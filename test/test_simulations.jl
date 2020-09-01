using Oceananigans.Simulations:
    stop, iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded, TimeStepWizard

function dependencies_added_correctly!(model, windowed_time_average, output_writer)

    model.clock.iteration = 0
    model.clock.time = 0.0
    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    push!(simulation.output_writers, output_writer)
    run!(simulation)

    return windowed_time_average ∈ values(simulation.diagnostics)
end

@testset "Simulations" begin
    @info "Testing simulations..."

    for arch in archs

        grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
        model = IncompressibleModel(architecture=arch, grid=grid)

        for Δt in (3, TimeStepWizard(Δt=5.0))
            model.clock.time = 0.0
            model.clock.iteration = 0

            simulation = Simulation(model, Δt=Δt, stop_iteration=1)

            # Just make sure we can construct a simulation without any errors.
            @test simulation isa Simulation

            @test iteration_limit_exceeded(simulation) == false
            @test stop(simulation) == false

            run!(simulation)

            # Just make sure run! executes without any errors.
            @test simulation isa Simulation

            # Some basic tests
            @test iteration_limit_exceeded(simulation) == true
            @test stop(simulation) == true

            t = Δt isa Number ? 3 : 5
            @test model.clock.time ≈ t
            @test model.clock.iteration == 1
            @test simulation.run_time > 0

            @test stop_time_exceeded(simulation) == false
            simulation.stop_time = 1e-12
            @test stop_time_exceeded(simulation) == true

            @test wall_time_limit_exceeded(simulation) == false
            simulation.wall_time_limit = 1e-12
            @test wall_time_limit_exceeded(simulation) == true

        end

        windowed_time_average = WindowedTimeAverage(model.velocities.u, time_window=2.0, time_interval=4.0)

        output = Dict(:time_average => windowed_time_average)

        jld2_output_writer = JLD2OutputWriter(model, output, time_interval=4.0, dir=".", prefix="test", force=true)

        @test dependencies_added_correctly!(model, windowed_time_average, jld2_output_writer)

        output = Dict("time_average" => windowed_time_average)
        attributes = Dict("time_average" => Dict("longname" => "A time average",  "units" => "arbitrary"))
        dimensions = Dict("time_average" => ("xF", "yC", "zC"))

        netcdf_output_writer =
            NetCDFOutputWriter(model, output, time_interval=4.0, filename="test.nc", with_halos=true,
                               output_attributes=attributes, dimensions=dimensions)

        @test dependencies_added_correctly!(model, windowed_time_average, netcdf_output_writer)
    end
end
