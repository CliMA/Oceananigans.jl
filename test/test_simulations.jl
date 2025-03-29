include("dependencies_for_runtests.jl")

using TimesDates: TimeDate

using Oceananigans.Models: erroring_NaNChecker!

import Oceananigans.Simulations: finalize!, initialize!

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
    @test all(called_at .≈ 0.0:schedule.interval:simulation.stop_time)

    # Test that minimum_relative_step is running correctly
    final_time = 1 + 1e-11
    model.clock.time = 0
    model.clock.iteration = 0
    simulation = Simulation(model, Δt=1, stop_time=final_time, minimum_relative_step=1e-10)
    run!(simulation)

    @test time(simulation) == final_time
    @test iteration(simulation) == 1

    model.clock.time = 0
    model.clock.iteration = 0
    simulation = Simulation(model, Δt=1, stop_time=3, align_time_step=true)
    simulation.callbacks[:tester] = Callback(sim -> nothing, TimeInterval(0.1))

    time_step!(simulation)
    @test time(simulation) == 0.1
    @test iteration(simulation) == 1

    time_step!(simulation)
    @test time(simulation) == 0.2
    @test iteration(simulation) == 2
    @test simulation.Δt == 1

    simulation.align_time_step = false
    time_step!(simulation)
    @test time(simulation) == 1.2
    @test iteration(simulation) == 3
    @test simulation.Δt == 1

    time_step!(simulation)
    @test time(simulation) == 2.2
    @test iteration(simulation) == 4
    @test simulation.Δt == 1

    simulation.stop_time = 5
    simulation.Δt = 2
    simulation.align_time_step = true
    time_step!(simulation, 1)
    @test !(simulation.align_time_step)
    @test time(simulation) == 3.2
    @test iteration(simulation) == 5
    @test simulation.Δt == 1

    return nothing
end

function run_simulation_date_tests(arch, start_time, stop_time, Δt)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

    clock = Clock(time=start_time)
    model = NonhydrostaticModel(; grid, clock, timestepper=:QuasiAdamsBashforth2)
    simulation = Simulation(model; Δt, stop_time)

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
    simulation = Simulation(model, Δt=1, stop_iteration=2)
    model.velocities.u[1, 1, 1] = NaN
    erroring && erroring_NaNChecker!(simulation)

    if erroring
        @test_throws ErrorException run!(simulation)
    else
        run!(simulation)
        @test model.clock.iteration == 1 # simulation stopped after one iteration
    end

    return nothing
end

@testset "Time step wizard" begin
    for arch in archs
        @info "Testing time step wizard [$(typeof(arch))]..."
        wall_time_step_wizard_tests(arch)
    end
end

mutable struct InitializedFinalized
    initialized :: Bool
    finalized :: Bool
end
(::InitializedFinalized)(sim) = nothing
function initialize!(infi::InitializedFinalized, sim)
    infi.initialized = true
    return nothing
end

function finalize!(infi::InitializedFinalized, sim)
    infi.finalized = true
    return nothing
end

@testset "Simulations" begin
    for arch in archs
        @info "Testing simulations [$(typeof(arch))]..."
        run_basic_simulation_tests(arch)

        # Test initialization for simulations started with iteration ≠ 0
        grid = RectilinearGrid(arch, size=(), topology=(Flat, Flat, Flat))
        model = NonhydrostaticModel(; grid)
        simulation = Simulation(model; Δt=1, stop_time=6)

        progress_message(sim) = @info string("Iter: ", iteration(sim), ", time: ", prettytime(sim))
        progress_cb = Callback(progress_message, TimeInterval(2))
        simulation.callbacks[:progress] = progress_cb

        model.clock.iteration = 1 # we want to start here for some reason
        run!(simulation)
        @test progress_cb.schedule.actuations == 3

        # Test initialize! and finalize!
        model = NonhydrostaticModel(; grid)
        simulation = Simulation(model; Δt=1, stop_time=6)
        infi = InitializedFinalized(false, false)
        add_callback!(simulation, infi, IterationInterval(1))
        @test !(infi.initialized)
        @test !(infi.finalized)
        time_step!(simulation) # should initialize
        @test infi.initialized
        @test !(infi.finalized)
        run!(simulation) # should finalize
        @test infi.initialized
        @test infi.finalized

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
