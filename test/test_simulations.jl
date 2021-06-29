using Oceananigans.Simulations:
    stop, iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded,
    TimeStepWizard, update_Δt!

function run_time_step_wizard_tests(arch)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch, grid=grid)

    Δx = grid.Δx
    CFL = 0.45
    u₀ = 7
    Δt = 2.5
    model.velocities.u[1, 1, 1] = u₀

    wizard = TimeStepWizard(cfl=CFL, Δt=Δt, max_change=Inf, min_change=0)
    update_Δt!(wizard, model)
    @test wizard.Δt ≈ CFL * Δx / u₀

    wizard = TimeStepWizard(cfl=CFL, Δt=Δt, max_change=Inf, min_change=0.75)
    update_Δt!(wizard, model)
    @test wizard.Δt ≈ 0.75Δt

    wizard = TimeStepWizard(cfl=CFL, Δt=Δt, max_change=Inf, min_change=0, min_Δt=1.99)
    update_Δt!(wizard, model)
    @test wizard.Δt ≈ 1.99

    model.velocities.u[1, 1, 1] = u₀/100

    wizard = TimeStepWizard(cfl=CFL, Δt=Δt, max_change=1.1, min_change=0)
    update_Δt!(wizard, model)
    @test wizard.Δt ≈ 1.1Δt

    wizard = TimeStepWizard(cfl=CFL, Δt=Δt, max_change=Inf, min_change=0, max_Δt=3.99)
    update_Δt!(wizard, model)
    @test wizard.Δt ≈ 3.99


    grid_stretched = VerticallyStretchedRectilinearGrid(size=(1,1,1), x=(0,1), y=(0,1), z_faces=z->z, 
                                                        halo=(1,1,1), architecture=arch)
    model = IncompressibleModel(architecture=arch, grid=grid_stretched)

    Δx = grid_stretched.Δx
    CFL = 0.45
    u₀ = 7
    Δt = 2.5
    model.velocities.u[1, 1, 1] = u₀

    CUDA.allowscalar(false)
    wizard = TimeStepWizard(cfl=CFL, Δt=Δt, max_change=Inf, min_change=0)
    update_Δt!(wizard, model)
    @test wizard.Δt ≈ CFL * Δx / u₀
    CUDA.allowscalar(true)

    return nothing
end

function run_basic_simulation_tests(arch, Δt)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch, grid=grid)

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

    # Test that simulation stops at `stop_iteration`.
    model = IncompressibleModel(architecture=arch, grid=grid)
    simulation = Simulation(model, Δt=Δt, stop_iteration=3, iteration_interval=88)
    run!(simulation)

    @test simulation.model.clock.iteration == 3

    # Test that simulation stops at `stop_time`.
    model = IncompressibleModel(architecture=arch, grid=grid)
    simulation = Simulation(model, Δt=Δt, stop_time=20.20, iteration_interval=123)
    run!(simulation)

    @test simulation.model.clock.time ≈ 20.20

    return nothing
end

function run_simulation_date_tests(arch, start_time, stop_time, Δt)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))

    clock = Clock(time=start_time)
    model = IncompressibleModel(architecture=arch, grid=grid, clock=clock)

    simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

    @test model.clock.time == start_time
    @test simulation.stop_time == stop_time

    run!(simulation)

    @test model.clock.time == stop_time
    @test simulation.stop_time == stop_time

    return nothing
end

@testset "Time step wizard" begin
    for arch in archs
        @info "Testing time step wizard [$(typeof(arch))]..."
        run_time_step_wizard_tests(arch)
    end
end

@testset "Simulations" begin
    for arch in archs
        @info "Testing simulations [$(typeof(arch))]..."
        for Δt in (3, TimeStepWizard(Δt=5.0))
            run_basic_simulation_tests(arch, Δt)
        end

        @info "Testing simulations with DateTime [$(typeof(arch))]..."
        run_simulation_date_tests(arch, 0.0, 1.0, 0.3)
        run_simulation_date_tests(arch, DateTime(2020), DateTime(2021), 100days)
        run_simulation_date_tests(arch, TimeDate(2020), TimeDate(2021), 100days)
    end
end
