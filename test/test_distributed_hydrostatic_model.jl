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
using Oceananigans.DistributedComputations: partition, all_reduce, cpu_architecture, reconstruct_global_grid

function Δ_min(grid) 
    Δx_min = minimum_xspacing(grid, Center(), Center(), Center())
    Δy_min = minimum_yspacing(grid, Center(), Center(), Center())
    return min(Δx_min, Δy_min)
end

function test_model_equality(test_model, true_model)
    CUDA.@allowscalar begin
        test_model_fields = prognostic_fields(test_model)
        true_model_fields = prognostic_fields(true_model)
        field_names = keys(test_model_fields)

        for name in field_names
            @test all(test_model_fields[name].data .≈ true_model_fields[name].data)

            if test_model.timestepper isa QuasiAdamsBashforth2TimeStepper
                if name ∈ keys(test_model.timestepper.Gⁿ)
                    @test all(test_model.timestepper.Gⁿ[name].data .≈ true_model.timestepper.Gⁿ[name].data)
                    @test all(test_model.timestepper.G⁻[name].data .≈ true_model.timestepper.G⁻[name].data)
                end
            end
        end
    end

    return nothing
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
    @testset "Distributed solid body rotation [$arch]" begin
        @info "Testing solid body rotation on $arch..."
        underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, 1),
                                                halo = (4, 4, 4),
                                                latitude = (-80, 80),
                                                longitude = (-160, 160),
                                                z = (-1, 0),
                                                radius = 1,
                                                topology=(Bounded, Bounded, Bounded))

        bottom(λ, φ) = -30 < λ < 30 && -40 < φ < 20 ? 0 : - 1

        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))
        immersed_active_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom); active_cells_map = true)

        global_underlying_grid = reconstruct_global_grid(underlying_grid)
        global_immersed_grid   = ImmersedBoundaryGrid(global_underlying_grid, GridFittedBottom(bottom))

        for (grid, global_grid) in zip((underlying_grid, immersed_grid, immersed_active_grid),
                                       (global_underlying_grid, global_immersed_grid, global_immersed_grid))

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

            us = partition(us, cpu_arch, size(u))
            vs = partition(vs, cpu_arch, size(v))
            ws = partition(ws, cpu_arch, size(w))
            cs = partition(cs, cpu_arch, size(c))
            ηs = partition(ηs, cpu_arch, size(η))

            atol = eps(eltype(grid))
            rtol = sqrt(eps(eltype(grid)))

            @test all(isapprox(u, us; atol, rtol))
            @test all(isapprox(v, vs; atol, rtol))
            @test all(isapprox(w, ws; atol, rtol))
            @test all(isapprox(c, cs; atol, rtol))
            @test all(isapprox(η, ηs; atol, rtol))
        end
    end

    @testset "Distributed checkpointing" begin
        # Create and run "true model"
        Nx, Ny, Nz = 16, 16, 4
        Lx, Ly, Lz = 1, 1, 1

        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), x=(-10, 10), y=(-10, 10), z=(-1, 0))
        closure = ScalarDiffusivity(ν=1e-2, κ=1e-2)
        true_model = HydrostaticFreeSurfaceModel(; grid, free_surface, closure, buoyancy=nothing, tracers=())
        test_model = deepcopy(true_model)

        ηᵢ(x, y, z) = 1e-1 * exp(-x^2 - y^2)
        ϵᵢ(x, y, z) = 1e-6 * randn()
        set!(true_model, η=ηᵢ, u=ϵᵢ, v=ϵᵢ)

        Δt = 1e-6
        true_simulation = Simulation(true_model, Δt=Δt, stop_iteration=5)
        checkpointer = Checkpointer(true_model, schedule=IterationInterval(5), overwrite_existing=true)
        push!(true_simulation.output_writers, checkpointer)
        run!(true_simulation) # for 5 iterations
        checkpointed_model = deepcopy(true_simulation.model)

        true_simulation.stop_iteration = 9
        run!(true_simulation) # for 4 more iterations

        #####
        ##### Test `set!(model, checkpoint_file)`
        #####

        rank = arch.local_rank
        set!(test_model, "checkpoint_$(rank)_iteration5.jld2")

        @test test_model.clock.iteration == checkpointed_model.clock.iteration
        @test test_model.clock.time == checkpointed_model.clock.time
        test_model_equality(test_model, checkpointed_model)

        # This only applies to QuasiAdamsBashforthTimeStepper:
        @test test_model.clock.last_Δt == checkpointed_model.clock.last_Δt

        #####
        ##### Test pickup from explicit checkpoint path
        #####

        test_simulation = Simulation(test_model, Δt=Δt, stop_iteration=9)

        # Pickup from explicit checkpoint path
        run!(test_simulation, pickup="checkpoint_$(rank)_iteration0.jld2")

        @info "Testing model equality when running with pickup=checkpoint_iteration0.jld2."
        @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
        @test test_simulation.model.clock.time == true_simulation.model.clock.time
        test_model_equality(test_model, true_model)

        run!(test_simulation, pickup="checkpoint_$(rank)_iteration5.jld2")
        @info "Testing model equality when running with pickup=checkpoint_iteration5.jld2."

        @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
        @test test_simulation.model.clock.time == true_simulation.model.clock.time
        test_model_equality(test_model, true_model)

        #####
        ##### Test `run!(sim, pickup=true)
        #####

        # Pickup using existing checkpointer
        test_simulation.output_writers[:checkpointer] =
            Checkpointer(test_model, schedule=IterationInterval(5), overwrite_existing=true)

        run!(test_simulation, pickup=true)
        @info "    Testing model equality when running with pickup=true."

        @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
        @test test_simulation.model.clock.time == true_simulation.model.clock.time
        test_model_equality(test_model, true_model)

        run!(test_simulation, pickup=0)
        @info "    Testing model equality when running with pickup=0."

        @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
        @test test_simulation.model.clock.time == true_simulation.model.clock.time
        test_model_equality(test_model, true_model)

        run!(test_simulation, pickup=5)
        @info "    Testing model equality when running with pickup=5."

        @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
        @test test_simulation.model.clock.time == true_simulation.model.clock.time
        test_model_equality(test_model, true_model)

        rm("checkpoint_$(rank)_iteration0.jld2", force=true)
        rm("checkpoint_$(rank)_iteration5.jld2", force=true)
    end
end

