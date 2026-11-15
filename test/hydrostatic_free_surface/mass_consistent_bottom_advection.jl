include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: inactive_node, znode
using Oceananigans.Operators: volume

# Potential energy of the same water sorted by buoyancy into the model's cells, densest lowest
function background_potential_energy(model)
    grid = on_architecture(CPU(), model.grid)
    b = on_architecture(CPU(), model.tracers.b)
    C = Center()
    Nx, Ny, Nz = size(grid)
    cells = [(i, j, k) for i in 1:Nx, j in 1:Ny, k in 1:Nz if !inactive_node(i, j, k, grid, C, C, C)]
    bs = [b[i, j, k] for (i, j, k) in cells]
    zs = [znode(i, j, k, grid, C, C, C) for (i, j, k) in cells]
    Vs = [volume(i, j, k, grid, C, C, C) for (i, j, k) in cells]

    lowest, densest = sortperm(zs), sortperm(bs)
    energy, n, room = 0.0, 1, Vs[lowest[1]]
    for m in densest
        left = Vs[m]
        while left > 1e-12 * Vs[m]
            taken = min(left, room)
            energy -= bs[m] * zs[lowest[n]] * taken
            left -= taken
            room -= taken
            if room ≤ 1e-12 * Vs[lowest[n]] && n < length(lowest)
                n += 1
                room = Vs[lowest[n]]
            end
        end
    end
    return energy
end

# A current crossing a plateau whose bottom cell is 20 % wet with partial cells, beside a full cell over the basin.
# The flow is unforced, so the rise of the background potential energy over 10 days is numerical mixing.
function plateau_channel_mixing(arch, direction, Bottom)
    N, Nz, Δ = 32, 20, 15e3
    basin, plateau = 2837, 1863
    on_plateau(s) = 12Δ ≤ s < 20Δ
    bottom(s) = on_plateau(s) ? -plateau : -basin
    u₀(s, z) = 0.1 * basin / (on_plateau(s) ? plateau : basin)
    b₀(s, z) = 0.01 * exp(z / 1000)

    z = ExponentialDiscretization(Nz, -5000, 0; scale = 1250, mutable = true)
    topology = direction == :x ? (Periodic, Flat, Bounded) : (Flat, Periodic, Bounded)
    extent = direction == :x ? (; x = (0, N * Δ)) : (; y = (0, N * Δ))
    underlying_grid = RectilinearGrid(arch; size = (N, Nz), z, halo = (7, 7), topology, extent...)
    grid = ImmersedBoundaryGrid(underlying_grid, Bottom(bottom))

    model = HydrostaticFreeSurfaceModel(grid; momentum_advection = WENOVectorInvariant(), tracer_advection = WENO(order = 5),
                                        buoyancy = BuoyancyTracer(), tracers = :b,
                                        free_surface = SplitExplicitFreeSurface(grid; substeps = 30))

    direction == :x ? set!(model; u = u₀, b = b₀) : set!(model; v = u₀, b = b₀)

    initial_energy = background_potential_energy(model)
    for _ in 1:1440
        time_step!(model, 600)
    end
    return background_potential_energy(model) - initial_energy
end

@testset "Mass-consistent momentum advection at bottom-cell thickness jumps [$(typeof(arch))]" for arch in archs
    whole = plateau_channel_mixing(arch, :x, GridFittedBottom)
    partial = plateau_channel_mixing(arch, :x, PartialCellBottom)
    partial_in_y = plateau_channel_mixing(arch, :y, PartialCellBottom)

    @test partial / whole < 1.3
    @test partial_in_y ≈ partial
end
