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
    end
end
