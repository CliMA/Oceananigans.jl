using Oceananigans.Simulations:
    stop, iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded

@testset "Simulations" begin
    @info "Testing simulations..."

    for arch in archs, Δt in (3, TimeStepWizard(Δt=5.0))
        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
        model = IncompressibleModel(architecture=arch, grid=grid)
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
end
