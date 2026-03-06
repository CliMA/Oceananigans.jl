# MWE: Reactant MLIR pass failure on periodic halo-filling kernels
#
# Reproduces the MLIR optimization pass bug without Oceananigans.
# Passes with H=1, fails with H=3 on Linux x64 CI.
#
# The kernel copies data between halo and interior regions of a 3D OffsetArray,
# mimicking periodic boundary conditions in Oceananigans.

using KernelAbstractions
using Reactant
using CUDA
using OffsetArrays

const ReactantKAExt = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt)
const RBackend = ReactantKAExt.ReactantBackend

struct Grid
    Nx::Int
    Ny::Int
    Nz::Int
    Hx::Int
    Hy::Int
    Hz::Int
end

@kernel function fill_periodic_west_and_east!(c, grid)
    j, k = @index(Global, NTuple)
    H = grid.Hx
    N = grid.Nx
    @inbounds for i = 1:H
        parent(c)[i, j, k]     = parent(c)[N+i, j, k]
        parent(c)[N+H+i, j, k] = parent(c)[H+i, j, k]
    end
end

@kernel function fill_periodic_south_and_north!(c, grid)
    i, k = @index(Global, NTuple)
    H = grid.Hy
    N = grid.Ny
    @inbounds for j = 1:H
        parent(c)[i, j, k]     = parent(c)[i, N+j, k]
        parent(c)[i, N+H+j, k] = parent(c)[i, H+j, k]
    end
end

@kernel function fill_periodic_bottom_and_top!(c, grid)
    i, j = @index(Global, NTuple)
    H = grid.Hz
    N = grid.Nz
    @inbounds for k = 1:H
        parent(c)[i, j, k]     = parent(c)[i, j, N+k]
        parent(c)[i, j, N+H+k] = parent(c)[i, j, H+k]
    end
end

function fill_periodic_halos!(c, grid)
    Sx = grid.Nx + 2grid.Hx
    Sy = grid.Ny + 2grid.Hy
    Sz = grid.Nz + 2grid.Hz

    we_kernel = fill_periodic_west_and_east!(RBackend(), (16, 16), (Sy, Sz))
    sn_kernel = fill_periodic_south_and_north!(RBackend(), (16, 16), (Sx, Sz))
    bt_kernel = fill_periodic_bottom_and_top!(RBackend(), (16, 16), (Sx, Sy))

    we_kernel(c, grid)
    sn_kernel(c, grid)
    bt_kernel(c, grid)
    return nothing
end

N, H = 4, 3
grid = Grid(N, N, N, H, H, H)
total = N + 2H

raw = Reactant.to_rarray(randn(total, total, total))
c = OffsetArray(raw, -H:N+H-1, -H:N+H-1, -H:N+H-1)

println("Compiling fill_periodic_halos! (H=$H)...")
compiled! = @compile raise=true raise_first=true fill_periodic_halos!(c, grid)
println("Running...")
compiled!(c, grid)
println("Success!")
