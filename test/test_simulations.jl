include("dependencies_for_runtests.jl")

using Oceananigans.Simulations:
    stop_iteration_exceeded, stop_time_exceeded, wall_time_limit_exceeded,
    TimeStepWizard, new_time_step, reset!

using Dates: DateTime

function wall_time_step_wizard_tests(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    Δx = grid.Δxᶜᵃᵃ

    model = NonhydrostaticModel(grid=grid)

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


    model = NonhydrostaticModel(grid=grid, closure=ScalarDiffusivity(ν=1))
    diff_CFL = 0.45

    wizard = TimeStepWizard(cfl=Inf, diffusive_cfl=diff_CFL, max_change=Inf, min_change=0)
    Δt = new_time_step(Δt, wizard, model)
    @test Δt ≈ diff_CFL * Δx^2 / model.closure.ν


    grid_stretched = RectilinearGrid(arch, 
                                    size = (1, 1, 1),
                                     x = (0, 1),
                                     y = (0, 1),
                                     z = z -> z, 
                                     halo = (1, 1, 1))

    model = NonhydrostaticModel(grid=grid_stretched)

    Δx = grid_stretched.Δxᶜᵃᵃ
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
    grid  = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid)
    simulation = Simulation(model, Δt=3, stop_iteration=1)

    # Just make sure we can construct a simulation without any errors.
    @test simulation isa Simulation

    simulation.running = true
    stop_iteration_exceeded(simulation)
    @test simulation.running

    run!(simulation)

    # Just make sure run! executes without any errors.
    @test simulation isa Simulation

    # Some basic tests
    simulation.running = true
    stop_iteration_exceeded(simulation)
    @test !(simulation.running)

    @test model.clock.time ≈ simulation.Δt
    @test model.clock.iteration == 1
    @test simulation.run_wall_time > 0

    simulation.running = true
    stop_time_exceeded(simulation)
    @test simulation.running

    simulation.running = true
    simulation.stop_time = 1e-12 # less than the current time.
    stop_time_exceeded(simulation)
    @test !(simulation.running)

    simulation.running = true
    wall_time_limit_exceeded(simulation)
    @test simulation.running

    simulation.running = true
    simulation.wall_time_limit = 1e-12
    wall_time_limit_exceeded(simulation)
    @test !(simulation.running)

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
    capture_call_time(sim, data) = push!(data, sim.model.clock.time)
    simulation.callbacks[:tester] = Callback(capture_call_time, schedule, parameters=called_at)
    run!(simulation)

    @show called_at
    @test all(called_at .≈ 0.0:schedule.interval:simulation.stop_time)

    return nothing
end

function run_simulation_date_tests(arch, start_time, stop_time, Δt)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

    clock = Clock(time=start_time)
    model = NonhydrostaticModel(grid=grid, clock=clock)
    simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

    @test model.clock.time == start_time
    @test simulation.stop_time == stop_time

    run!(simulation)

    @test model.clock.time == stop_time
    @test simulation.stop_time == stop_time

    return nothing
end

function run_nan_checker_test(arch; erroring)
    grid = RectilinearGrid(arch, size=(4, 2, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid)
    simulation = Simulation(model, Δt=1, stop_iteration=1)
    model.velocities.u[1, 1, 1] = NaN
    erroring && erroring_NaNChecker!(simulation)

    if erroring
        @test_throws ErrorException run!(simulation)
    else
        run!(simulation)
        @test model.clock.iteration == 0 # simulation did not run
    end

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

        @testset "NaN Checker [$(typeof(arch))]" begin
            @info "  Testing NaN Checker [$(typeof(arch))]..."
            run_nan_checker_test(arch, erroring=true)
            run_nan_checker_test(arch, erroring=false)
        end

        @info "Testing simulations with DateTime [$(typeof(arch))]..."
        run_simulation_date_tests(arch, 0.0, 1.0, 0.3)
        run_simulation_date_tests(arch, DateTime(2020), DateTime(2021), 100days)
        run_simulation_date_tests(arch, TimeDate(2020), TimeDate(2021), 100days)
    end
end
