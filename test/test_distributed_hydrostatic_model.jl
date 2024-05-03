include("dependencies_for_runtests.jl")

using MPI

# # Distributed model tests
#
# These tests are meant to be run on 4 ranks. This script may be run
# stand-alone (outside the test environment) via
#
# mpiexec -n 4 julia --project test_distributed_models.jl
#
# provided that a few packages (like TimesDates.jl) are in your global environment.
#
# Another possibility is to use tmpi ():
#
# tmpi 4 julia --project
#
# then later:
# 
# julia> include("test_distributed_models.jl")
#
# When running the tests this way, uncomment the following line

MPI.Initialized() || MPI.Init()

# to initialize MPI.

using Oceananigans.Operators: hack_cosd
using Oceananigans.DistributedComputations: partition_global_array, all_reduce, cpu_architecture, reconstruct_global_grid

function Δ_min(grid) 
    Δx_min = minimum_xspacing(grid, Center(), Center(), Center())
    Δy_min = minimum_yspacing(grid, Center(), Center(), Center())
    return min(Δx_min, Δy_min)
end

@inline Gaussian(x, y, L) = exp(-(x^2 + y^2) / L^2)

function solid_body_rotation_test(grid)

    free_surface = SplitExplicitFreeSurface(grid; substeps = 5, gravitational_acceleration = 1)
    coriolis     = HydrostaticSphericalCoriolis(rotation_rate = 1)

    model = HydrostaticFreeSurfaceModel(; grid,
                                        momentum_advection = VectorInvariant(),
                                        free_surface = free_surface,
                                        coriolis = coriolis,
                                        tracers = :c,
                                        tracer_advection = WENO(),
                                        buoyancy = nothing,
                                        closure = nothing)

    g = model.free_surface.gravitational_acceleration
    R = grid.radius
    Ω = model.coriolis.rotation_rate

    uᵢ(λ, φ, z) = 0.1 * cosd(φ) * sind(λ)
    ηᵢ(λ, φ, z) = (R * Ω * 0.1 + 0.1^2 / 2) * sind(φ)^2 / g * sind(λ)

    # Gaussian leads to values with O(1e-60),
    # too small for repetible testing. We cap it at 0.1
    cᵢ(λ, φ, z) = max(Gaussian(λ, φ - 5, 10), 0.1)
    vᵢ(λ, φ, z) = 0.1

    set!(model, u=uᵢ, η=ηᵢ, c=cᵢ)

    @show Δt_local = 0.1 * Δ_min(grid) / sqrt(g * grid.Lz) 
    @show Δt = all_reduce(min, Δt_local, architecture(grid))

    simulation = Simulation(model; Δt, stop_iteration = 10)
    run!(simulation)

    return merge(model.velocities, model.tracers, (; η = model.free_surface.η))
end

Nx = 32
Ny = 32

for arch in archs
    @testset "Testing distributed solid body rotation" begin
        underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, 1),
                                                halo = (4, 4, 4),
                                                latitude = (-80, 80),
                                                longitude = (-160, 160),
                                                z = (-1, 0),
                                                radius = 1,
                                                topology=(Bounded, Bounded, Bounded))

        bottom(λ, φ) = -30 < λ < 30 && -40 < φ < 20 ? 0 : - 1

        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

        global_underlying_grid = reconstruct_global_grid(underlying_grid)
        global_immersed_grid   = ImmersedBoundaryGrid(global_underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

        for (grid, global_grid) in zip((underlying_grid, immersed_grid), (global_underlying_grid, global_immersed_grid))

            # "s" for "serial" computation
            us, vs, ws, cs, ηs = solid_body_rotation_test(global_grid)

            us = interior(on_architecture(CPU(), us))
            vs = interior(on_architecture(CPU(), vs))
            ws = interior(on_architecture(CPU(), ws))
            cs = interior(on_architecture(CPU(), cs))
            ηs = interior(on_architecture(CPU(), ηs))

            @info "  Testing distributed solid body rotation with architecture $arch on $(typeof(grid).name.wrapper)"
            u, v, w, c, η = solid_body_rotation_test(grid)

            cpu_arch = cpu_architecture(arch)

            u = interior(on_architecture(cpu_arch, u))
            v = interior(on_architecture(cpu_arch, v))
            w = interior(on_architecture(cpu_arch, w))
            c = interior(on_architecture(cpu_arch, c))
            η = interior(on_architecture(cpu_arch, η))

            us = partition_global_array(cpu_arch, us, size(u))
            vs = partition_global_array(cpu_arch, vs, size(v))
            ws = partition_global_array(cpu_arch, ws, size(w))
            cs = partition_global_array(cpu_arch, cs, size(c))
            ηs = partition_global_array(cpu_arch, ηs, size(η))

            atol = eps(eltype(grid))
            rtol = sqrt(eps(eltype(grid)))

            @test all(isapprox(u, us; atol, rtol))
            @test all(isapprox(v, vs; atol, rtol))
            @test all(isapprox(w, ws; atol, rtol))
            @test all(isapprox(c, cs; atol, rtol))
            @test all(isapprox(η, ηs; atol, rtol))
        end
    end
end
