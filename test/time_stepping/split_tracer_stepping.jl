include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "split_tracer_stepping_test_utils.jl"))

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity

ridge(x₀, width, height, depth) = (x, y) -> - depth + height * exp(- ((x - x₀) / width)^2)

function static_basin_grids(arch)
    rectilinear = RectilinearGrid(arch; size = (16, 8, 6), halo = (4, 4, 4), x = (0, 32kilometers), y = (0, 16kilometers),
                                  z = (-60, 0), topology = (Bounded, Bounded, Bounded))

    latitude_longitude = LatitudeLongitudeGrid(arch; size = (16, 8, 6), halo = (4, 4, 4), longitude = (0, 0.32),
                                               latitude = (30, 30.16), z = (-60, 0))

    return (rectilinear,
            ImmersedBoundaryGrid(rectilinear, GridFittedBottom(ridge(16kilometers, 4kilometers, 30, 60))),
            ImmersedBoundaryGrid(latitude_longitude, GridFittedBottom(ridge(0.16, 0.04, 30, 60))))
end

@testset "Split tracer time stepping" begin
    for arch in archs
        grid = RectilinearGrid(arch; size = (8, 4, 4), halo = (4, 4, 4), x = (0, 16kilometers), y = (0, 8kilometers),
                               z = (-40, 0), topology = (Bounded, Bounded, Bounded))

        @testset "TracerTimeStepSplitting constructor and show [$(typeof(arch))]" begin
            @info "  Testing the TracerTimeStepSplitting constructor [$(typeof(arch))]..."
            splitting = TracerTimeStepSplitting(tracers = :c, ratio = 4)
            @test splitting.tracer_names == (:c,)
            @test splitting.ratio == 4
            @test isnothing(splitting.biogeochemistry_substeps)
            @test occursin("ratio: 4", sprint(show, splitting))

            @test_throws ArgumentError TracerTimeStepSplitting(tracers = :c, ratio = 0)
            @test_throws ArgumentError TracerTimeStepSplitting(tracers = :c, ratio = 2.5)
            @test_throws ArgumentError TracerTimeStepSplitting(tracers = :c, ratio = 2, biogeochemistry_substeps = 0)
            @test_throws ArgumentError TracerTimeStepSplitting(tracers = (), ratio = 2)

            splitting = TracerTimeStepSplitting(tracers = (:c, :d), ratio = 3)
            model = HydrostaticFreeSurfaceModel(grid; tracers = (:b, :c, :d), timestepper = :SplitRungeKutta3,
                                                tracer_time_step_splitting = splitting)
            @test keys(model.tracer_time_step_splitting.slow_tracers) == (:c, :d)
            @test keys(model.tracer_time_step_splitting.fast_tracers) == (:b,)
            @test keys(model.tracers) == (:b, :c, :d)
            @test occursin("tracer_time_step_splitting", sprint(show, model))

            unknown = TracerTimeStepSplitting(tracers = :q, ratio = 2)
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; tracers = :c, tracer_time_step_splitting = unknown)

            substeps_without_biogeochemistry = TracerTimeStepSplitting(tracers = :c, ratio = 2, biogeochemistry_substeps = 2)
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; tracers = :c, tracer_time_step_splitting = substeps_without_biogeochemistry)

            closure_tracer = TracerTimeStepSplitting(tracers = :e, ratio = 2)
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; tracers = :c, closure = CATKEVerticalDiffusivity(),
                                                                   tracer_time_step_splitting = closure_tracer)

            zstar_grid = RectilinearGrid(arch; size = (8, 4, 4), halo = (4, 4, 4), x = (0, 16kilometers), y = (0, 8kilometers),
                                         z = MutableVerticalDiscretization(collect(-40:10:0)), topology = (Bounded, Bounded, Bounded))
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(zstar_grid; tracers = :c, timestepper = :QuasiAdamsBashforth2,
                                                                   vertical_coordinate = ZStarCoordinate(),
                                                                   tracer_time_step_splitting = TracerTimeStepSplitting(tracers = :c, ratio = 2))
        end

        @testset "ratio = 1 matches the unsplit model in a steady flow [$(typeof(arch))]" begin
            @info "  Testing that ratio = 1 matches the unsplit model [$(typeof(arch))]..."
            split = overturning_model(grid; ratio = 1, period = Inf)
            unsplit = overturning_model(grid; ratio = nothing, period = Inf)

            for step in 1:4
                time_step!(split, 1minute)
                time_step!(unsplit, 1minute)
            end

            # The long step reconstructs the transport as (Δt ũ σ) / (Δt σ), so the match is to round-off rather than bitwise
            for name in (:c, :smooth)
                @test Array(interior(split.tracers[name])) ≈ Array(interior(unsplit.tracers[name])) rtol = 1e-12
            end
        end

        @testset "Slow tracers change only on long steps; the fast group is untouched [$(typeof(arch))]" begin
            @info "  Testing that slow tracers change only on long steps [$(typeof(arch))]..."
            for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
                ratio = 3
                split = overturning_model(grid; ratio, timestepper)
                unsplit = overturning_model(grid; ratio = nothing, timestepper)

                previous_c = Array(interior(split.tracers.c))
                changed_at = Int[]

                for step in 1:2ratio
                    time_step!(split, 1minute)
                    time_step!(unsplit, 1minute)
                    c = Array(interior(split.tracers.c))
                    c != previous_c && push!(changed_at, step)
                    previous_c = c
                end

                @test changed_at == [ratio, 2ratio]
                @test Array(parent(split.tracers.fast)) == Array(parent(unsplit.tracers.fast))
                @test Array(parent(split.velocities.u)) == Array(parent(unsplit.velocities.u))
                @test Array(parent(split.free_surface.displacement)) == Array(parent(unsplit.free_surface.displacement))
            end
        end

        @testset "Slow tracers are masked at the start of a cycle and by the long step only [$(typeof(arch))]" begin
            @info "  Testing immersed masking of slow tracers [$(typeof(arch))]..."
            immersed_grid = static_basin_grids(arch)[2]
            dry = .!active_cells(immersed_grid)
            model = overturning_model(immersed_grid; ratio = 3)
            c = model.tracers.constant
            dry_values(c) = Array(interior(c))[dry]

            parent(c) .= 1
            time_step!(model, 1minute)
            @test all(iszero, dry_values(c))

            parent(c) .= 1
            time_step!(model, 1minute)
            @test all(isone, dry_values(c))

            time_step!(model, 1minute)
            @test all(iszero, dry_values(c))
        end

        @testset "Static z uniform tracer and inventory conservation [$(typeof(arch))]" begin
            for basin_grid in static_basin_grids(arch), timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3), ratio in (1, 4)
                @info "  Testing static z conservation on $(summary(basin_grid)) with $timestepper and ratio $ratio [$(typeof(arch))]..."
                model = overturning_model(basin_grid; ratio, timestepper, speed = 2, period = 30minutes)
                active = active_cells(model.grid)
                inventory₀ = tracer_inventory(model.tracers.c)

                for step in 1:2ratio
                    time_step!(model, 1minute)
                end

                @test maximum_uniform_deviation(model.tracers.constant, 1, active) < 1e-12
                @test abs(tracer_inventory(model.tracers.c) - inventory₀) / inventory₀ < 1e-12
            end
        end

        zstar_grid = ImmersedBoundaryGrid(RectilinearGrid(arch; size = (16, 8, 6), halo = (4, 4, 4),
                                                          x = (0, 32kilometers), y = (0, 16kilometers),
                                                          z = MutableVerticalDiscretization(collect(range(-60, 0, length=7))),
                                                          topology = (Bounded, Bounded, Bounded)),
                                          GridFittedBottom(ridge(16kilometers, 4kilometers, 30, 60)))

        @testset "NPZD total nitrogen with sub-cycled sources [$(typeof(arch))]" begin
            @info "  Testing NPZD total nitrogen conservation with sub-cycled sources [$(typeof(arch))]..."
            ratio = 16
            Δt = 5minutes
            stiff_remineralization_rate = 1 / 1000seconds

            for substeps in (nothing, 4)
                model = npzd_model(zstar_grid; ratio, biogeochemistry_substeps = substeps, sinking_speed = 20 / day,
                                   remineralization_rate = stiff_remineralization_rate)
                total₀ = total_nitrogen(model)

                for step in 1:2ratio
                    time_step!(model, Δt)
                end

                tracers_are_bounded = all(name -> maximum(abs, Array(interior(model.tracers[name]))) < 50, (:N, :P, :Z, :D))

                if isnothing(substeps)
                    # r N Δt = 4.8 lies beyond the stability region of the Runge-Kutta step
                    @test !tracers_are_bounded
                else
                    @test tracers_are_bounded
                    @test abs(total_nitrogen(model) - total₀) / total₀ < 1e-12
                end
            end
        end

        @testset "Checkpointing mid-cycle and at the end of a cycle is bitwise [$(typeof(arch))]" begin
            @info "  Testing split tracer stepping checkpoint and restart [$(typeof(arch))]..."
            ratio = 4
            Δt = 5minutes
            closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ = 1e-3)

            build() = npzd_model(zstar_grid; ratio, biogeochemistry_substeps = 2, closure)

            continuous = build()
            for step in 1:3ratio
                time_step!(continuous, Δt)
            end

            for checkpoint_iteration in (ratio + 2, 2ratio)
                prefix = "split_tracer_checkpoint_$(checkpoint_iteration)"
                model = build()
                simulation = Simulation(model; Δt, stop_iteration = checkpoint_iteration)
                simulation.output_writers[:checkpointer] = Checkpointer(model; schedule = IterationInterval(checkpoint_iteration), prefix)
                run!(simulation)

                restarted = build()
                simulation = Simulation(restarted; Δt, stop_iteration = 3ratio)
                simulation.output_writers[:checkpointer] = Checkpointer(restarted; schedule = IterationInterval(10ratio), prefix)
                run!(simulation, pickup = true)

                for name in (:b, :N, :P, :Z, :D)
                    @test Array(parent(restarted.tracers[name])) == Array(parent(continuous.tracers[name]))
                end
                @test Array(parent(restarted.free_surface.displacement)) == Array(parent(continuous.free_surface.displacement))

                rm.(filter(startswith(prefix), readdir()); force = true)
            end
        end
    end
end
