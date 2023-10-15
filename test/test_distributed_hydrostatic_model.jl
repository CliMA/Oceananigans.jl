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

Gaussian(x, y, L) = exp(-(x^2 + y^2) / 2L^2)

function Δ_min(grid) 
    Δx_min = minimum_xspacing(grid, Center(), Center(), Center())
    Δy_min = minimum_yspacing(grid, Center(), Center(), Center())
    return min(Δx_min, Δy_min)
end

function solid_body_rotation_test(grid)

    free_surface = SplitExplicitFreeSurface(; substeps = 5, gravitational_acceleration = 1)
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
    # too small for repetible testing. We cap it at 1e-5.
    cᵢ(λ, φ, z) = max(Gaussian(λ, φ - 5, 10), 1e-5)

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
        grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, 1),
                                     halo = (3, 3, 3),
                                     latitude = (-80, 80),
                                     longitude = (-160, 160),
                                     z = (-1, 0),
                                     radius = 1,
                                     topology=(Bounded, Bounded, Bounded))

        global_grid = reconstruct_global_grid(grid)

        # "s" for "serial" computation
        us, vs, ws, cs, ηs = solid_body_rotation_test(global_grid)

        us = Array(interior(us))
        vs = Array(interior(vs))
        ws = Array(interior(ws))
        cs = Array(interior(cs))
        ηs = Array(interior(ηs))

        @info "  Testing distributed solid body rotation with architecture $arch"
        u, v, w, c, η = solid_body_rotation_test(grid)

        u = Array(interior(u))
        v = Array(interior(v))
        w = Array(interior(w))
        c = Array(interior(c))
        η = Array(interior(η))

        cpu_arch = cpu_architecture(arch)

        us = partition_global_array(cpu_arch, us, size(u))
        vs = partition_global_array(cpu_arch, vs, size(v))
        ws = partition_global_array(cpu_arch, ws, size(w))
        cs = partition_global_array(cpu_arch, cs, size(c))
        ηs = partition_global_array(cpu_arch, ηs, size(η))

        @test all(u .≈ us)
        @test all(v .≈ vs)
        @test all(w .≈ ws)
        @test all(c .≈ cs)
        @test all(η .≈ ηs)
    end
end
