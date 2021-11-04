using Test

using Oceananigans
using Oceananigans.Units
using Oceananigans.Simulations:
    stop, iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded,
    TimeStepWizard, new_time_step, reset!

using Dates: DateTime

include("utils_for_runtests.jl")

archs = test_architectures()

function wall_time_step_wizard_tests(arch)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(architecture=arch, grid=grid)

    Δx = grid.Δx
    CFL = 0.45
    u₀ = 7
    Δt = 2.5
    model.velocities.u[1, 1, 1] = u₀

    wizard = TimeStepWizard(cfl=CFL, max_change=Inf, min_change=0)
    Δt = new_time_step(Δt, wizard, model)
    @test Δt ≈ CFL * Δx / u₀

    wizard = TimeStepWizard(cfl=CFL, max_change=Inf, min_change=0.75)
    Δt = new_time_step(1.0, wizard, model)
    @test Δt ≈ 0.75

    wizard = TimeStepWizard(cfl=CFL, max_change=Inf, min_change=0, min_Δt=1.99)
    Δt = new_time_step(Δt, wizard, model)
    @test Δt ≈ 1.99

    model.velocities.u[1, 1, 1] = u₀/100

    wizard = TimeStepWizard(cfl=CFL, max_change=1.1, min_change=0)
    Δt = new_time_step(1.0, wizard, model)
    @test Δt ≈ 1.1

    wizard = TimeStepWizard(cfl=CFL, max_change=Inf, min_change=0, max_Δt=3.99)
    Δt = new_time_step(Δt, wizard, model)
    @test Δt ≈ 3.99

    grid_stretched = VerticallyStretchedRectilinearGrid(size = (1, 1, 1),
                                                        x = (0, 1),
                                                        y = (0, 1),
                                                        z_faces = z -> z, 
                                                        halo = (1, 1, 1),
                                                        architecture=arch)

    model = NonhydrostaticModel(architecture=arch, grid=grid_stretched)

    Δx = grid_stretched.Δx
    CFL = 0.45
    u₀ = 7
    Δt = 2.5
    model.velocities.u .= u₀

    wizard = TimeStepWizard(cfl=CFL, max_change=Inf, min_change=0)
    Δt = new_time_step(Δt, wizard, model)
    @test Δt ≈ CFL * Δx / u₀

    return nothing
end

function run_basic_simulation_tests(arch)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(architecture=arch, grid=grid)
    simulation = Simulation(model, Δt=3, stop_iteration=1)

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

    @test model.clock.time ≈ simulation.Δt
    @test model.clock.iteration == 1
    @test simulation.run_wall_time > 0

    @test stop_time_exceeded(simulation) == false
    simulation.stop_time = 1e-12
    @test stop_time_exceeded(simulation) == true

    @test wall_time_limit_exceeded(simulation) == false
    simulation.wall_time_limit = 1e-12
    @test wall_time_limit_exceeded(simulation) == true

    # Test that simulation stops at `stop_iteration`.
    reset!(simulation)
    simulation.stop_iteration = 3
    run!(simulation)

    @test simulation.model.clock.iteration == 3

    # Test that simulation stops at `stop_time`.
    reset!(simulation)
    simulation.stop_time = 20.20
    run!(simulation)

    @test simulation.model.clock.time ≈ 20.20

    # Test that we can run a simulation with TimeStepWizard
    reset!(simulation)
    simulation.stop_iteration = 2
  
    wizard = TimeStepWizard(cfl=0.1)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

    run!(simulation)

    @test simulation.callbacks[:wizard].func isa TimeStepWizard

    # Test time-step alignment with callbacks
    reset!(simulation)
    simulation.stop_time = 2.0
    simulation.Δt = 1.0

    called_at = Float64[]
    schedule = TimeInterval(0.31)
    capture_call_time(sim) = push!(called_at, sim.model.clock.time)
    simulation.callbacks[:tester] = Callback(capture_call_time, schedule)
    run!(simulation)

    @show called_at
    @test all(called_at .≈ 0.0:schedule.interval:simulation.stop_time)

    return nothing
end

function run_simulation_date_tests(arch, start_time, stop_time, Δt)
    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))

    clock = Clock(time=start_time)
    model = NonhydrostaticModel(architecture=arch, grid=grid, clock=clock)
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
        wall_time_step_wizard_tests(arch)
    end
end

@testset "Simulations" begin
    for arch in archs
        @info "Testing simulations [$(typeof(arch))]..."
        run_basic_simulation_tests(arch)

        @info "Testing simulations with DateTime [$(typeof(arch))]..."
        run_simulation_date_tests(arch, 0.0, 1.0, 0.3)
        run_simulation_date_tests(arch, DateTime(2020), DateTime(2021), 100days)
        run_simulation_date_tests(arch, TimeDate(2020), TimeDate(2021), 100days)
    end
end
