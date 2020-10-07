using Oceananigans.Grids: unpack_grid, AbstractGrid
using Oceananigans.BoundaryConditions: PBC, ZFBC

const PeriodicBC = PBC
const NoFluxBC = ZFBC

function generate_discrete_eigenvalues(grid, pressure_bcs)
    kx² = λi(grid, pressure_bcs.x.left)
    ky² = λj(grid, pressure_bcs.y.left)
    kz² = λk(grid, pressure_bcs.z.left)
    return kx², ky², kz²
end

"""
    λi(grid, ::PeriodicBC)

Return an Nx×1×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the x-dimension on `grid`.
"""
function λi(grid, ::PeriodicBC)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    is = reshape(1:Nx, Nx, 1, 1)
    return @. (2sin((is - 1) * π / Nx) / (Lx / Nx))^2
end

"""
    λi(grid, ::NoFluxBC)

Return an Nx×1×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the x-dimension on `grid`.
"""
function λi(grid, ::NoFluxBC)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    is = reshape(1:Nx, Nx, 1, 1)
    return @. (2sin((is - 1) * π / 2Nx) / (Lx / Nx))^2
end

"""
    λj(grid, ::PeriodicBC)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the y-dimension on `grid`.
"""
function λj(grid, ::PeriodicBC)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    return @. (2sin((js - 1) * π / Ny) / (Ly / Ny))^2
end

"""
    λj(grid, ::NoFluxBC)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the y-dimension on `grid`.
"""
function λj(grid, ::NoFluxBC)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    return @. (2sin((js - 1) * π / 2Ny) / (Ly / Ny))^2
end

"""
    λk(grid, ::PeriodicBC)

Return an 1×1×Nz array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the z-dimension on `grid`.
"""
function λk(grid, ::PeriodicBC)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    ks = reshape(1:Nz, 1, 1, Nz)
    return @. (2sin((ks - 1) * π / Nz) / (Lz / Nz))^2
end

"""
    λk(grid, ::NoFluxBC)

Return an 1×1×Nz array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the z-dimension on `grid`.
"""
function λk(grid, ::NoFluxBC)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    ks = reshape(1:Nz, 1, 1, Nz)
    return @. (2sin((ks - 1) * π / 2Nz) / (Lz / Nz))^2
end

# For Flat dimensions
λi(grid::AbstractGrid{FT, <:Flat}, ::Nothing) where FT = reshape([zero(FT)], 1, 1, 1)
λj(grid::AbstractGrid{FT, TX, <:Flat}, ::Nothing) where {FT, TX} = reshape([zero(FT)], 1, 1, 1)
λk(grid::AbstractGrid{FT, TX, TY, <:Flat}, ::Nothing) where {FT, TX, TY} = reshape([zero(FT)], 1, 1, 1)
