include("dependencies_for_runtests.jl")

import Dates
using Dates: DateTime

const HAS_TIMEDATE = try
    @eval using TimesDates: TimeDate
    true
catch
    false
end

using Oceananigans
using Oceananigans: set!
using Oceananigans.Architectures: CPU
using Oceananigans.Simulations: Simulation, run!, Callback, conjure_time_step_wizard!
using Oceananigans.OutputWriters: JLD2Writer
using Oceananigans.Units: hour, Time
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Utils: TimeInterval, SpecifiedTimes, schedule_aligned_time_step, IterationInterval, period_to_seconds
import Oceananigans: initialize!

struct DummyModel
    clock::Clock
end

function run_forcing_simulation(arch, FT, start_time; Δt, iterations)
    grid = RectilinearGrid(arch, FT; size=(1, 1, 1), extent=(FT(1), FT(1), FT(1)))

    time_points = [start_time + Dates.Hour(n) for n in 0:iterations]
    u_forcing = FieldTimeSeries{Face, Center, Center}(grid, time_points)

    for n in 1:length(time_points)
        set!(u_forcing[n], (x, y, z) -> FT(n - 1))
    end

    clock = Clock(time=start_time)
    model = HydrostaticFreeSurfaceModel(; grid, clock, forcing=(; u=u_forcing))
    distant_stop_time = start_time + Dates.Hour(iterations + 10)
    simulation = Simulation(model;
                            Δt=Δt,
                            stop_iteration=iterations,
                            stop_time=distant_stop_time,
                            align_time_step=false,
                            verbose=false)

    forcing_history = FT[]
    time_history = typeof(start_time)[]
    Δt_history = []

    push!(forcing_history, u_forcing[1, 1, 1, Time(model.clock.time)])
    push!(time_history, model.clock.time)

    simulation.callbacks[:capture] = Callback(IterationInterval(1)) do sim
        sim.model.clock.iteration == 0 && return
        push!(forcing_history, u_forcing[1, 1, 1, Time(sim.model.clock.time)])
        push!(time_history, sim.model.clock.time)
        push!(Δt_history, sim.model.clock.last_Δt)
    end

    run!(simulation)

    return forcing_history, time_history, Δt_history
end

function time_interval_schedule_checks(start_time)
    schedule = TimeInterval(Dates.Hour(1))
    clock = Clock(time=start_time)
    model = DummyModel(clock)

    initialize!(schedule, model)
    @test !schedule(model)

    # Alignment trims the step to the next hourly mark
    @test schedule_aligned_time_step(schedule, clock, 2hour) == hour

    tick!(clock, hour)
    @test schedule(model)
    @test !schedule(model) # triggers once per actuation

    # Mid-interval alignment
    schedule = TimeInterval(Dates.Hour(1))
    initialize!(schedule, DummyModel(Clock(time=start_time)))
    mid_clock = Clock(time=start_time + Dates.Minute(30))
    aligned = schedule_aligned_time_step(schedule, mid_clock, hour)
    @test aligned ≈ hour / 2 atol=1e-12

    return true
end

function specified_times_schedule_checks(start_time)
    schedule = SpecifiedTimes(start_time + Dates.Hour(1), start_time + Dates.Hour(3))
    clock = Clock(time=start_time)
    model = DummyModel(clock)

    initialize!(schedule, model)
    @test !schedule(model)

    # Alignment to first specified time
    @test schedule_aligned_time_step(schedule, clock, 2hour) == hour

    tick!(clock, hour)
    @test schedule(model)
    @test !schedule(model)

    # Alignment to final specified time after intermediate advance
    tick!(clock, hour)
    aligned = schedule_aligned_time_step(schedule, clock, hour)
    @test aligned == hour

    tick!(clock, aligned)
    @test schedule(model)
    @test !schedule(model)

    # No more specified times => alignment returns provided Δt
    @test schedule_aligned_time_step(schedule, clock, hour) == hour

    return true
end

function numeric_time_interval_schedule_checks(FT)
    schedule = TimeInterval(FT(2))
    clock = Clock(time=zero(FT))
    model = DummyModel(clock)

    initialize!(schedule, model)
    @test !schedule(model)

    Δt = FT(3)
    @test schedule_aligned_time_step(schedule, clock, Δt) == FT(2)

    tick!(clock, FT(2))
    @test schedule(model)
    @test !schedule(model)

    tick!(clock, FT(1))
    aligned = schedule_aligned_time_step(schedule, clock, FT(5))
    @test aligned == FT(1)

    tick!(clock, aligned)
    @test schedule(model)
    @test !schedule(model)

    return true
end

function numeric_specified_times_schedule_checks(FT)
    schedule = SpecifiedTimes(FT(1), FT(3))
    clock = Clock(time=zero(FT))
    model = DummyModel(clock)

    initialize!(schedule, model)
    @test !schedule(model)

    Δt = FT(5)
    @test schedule_aligned_time_step(schedule, clock, Δt) == FT(1)

    tick!(clock, FT(1))
    @test schedule(model)
    @test !schedule(model)

    tick!(clock, FT(1))
    aligned = schedule_aligned_time_step(schedule, clock, FT(2))
    @test aligned == FT(1)

    tick!(clock, aligned)
    @test schedule(model)
    @test !schedule(model)

    @test schedule_aligned_time_step(schedule, clock, FT(1)) == FT(1)

    return true
end

@testset "DateTime clocks" begin
    @info "Testing DateTime clock behavior..."

    clock_specs = [(:DateTime, DateTime)]
    if HAS_TIMEDATE
        push!(clock_specs, (:TimeDate, TimeDate))
    end

    for arch in archs, FT in float_types
        arch_type = typeof(arch)
        for (clock_label, ctor) in clock_specs
            start_time = ctor(2020, 1, 1)
            iterations = 3
            expected_times = [start_time + Dates.Hour(n) for n in 0:iterations]
            expected_forcing = FT.(0:iterations)

            step_specs = [(:numeric_seconds, convert(FT, hour)),
                          (:calendar_period, Dates.Hour(1))]

            for (step_label, Δt_spec) in step_specs
                forcing_history, time_history, Δt_history = run_forcing_simulation(arch, FT, start_time;
                                                                                   Δt=Δt_spec,
                                                                                   iterations=iterations)

                expected_dt = 3600.0

                @testset "Hydrostatic $(clock_label) forcing [$arch_type, $FT, $(step_label)]" begin
                    @test length(time_history) == iterations + 1
                    @test time_history == expected_times
                    @test forcing_history == expected_forcing
                    @test length(Δt_history) == iterations
                    @test all(isapprox.(Δt_history, expected_dt; atol=1e-6))
                end
            end

            if arch isa Oceananigans.Architectures.CPU
                # Simulation output + FieldTimeSeries reload
                @testset "Simulation output reload [$arch_type, $FT, $(clock_label)]" begin
                    path = tempname()
                    filename = path * "_output"
                    grid = RectilinearGrid(arch, FT; size=(1, 1, 1), extent=(FT(1), FT(1), FT(1)))

                    forcing_times = [start_time + Dates.Hour(n) for n in 0:iterations]
                    u_forcing = FieldTimeSeries{Face, Center, Center}(grid, forcing_times)
                    for n in 1:length(forcing_times)
                        set!(u_forcing[n], (x, y, z) -> FT(n - 1))
                    end

                    clock = Clock(time=start_time)
                    model = HydrostaticFreeSurfaceModel(; grid, clock, forcing=(; u=u_forcing))
                    distant_stop_time = start_time + Dates.Hour(iterations + 10)
                    simulation = Simulation(model;
                                            Δt=convert(FT, hour),
                                            stop_iteration=iterations,
                                            stop_time=distant_stop_time,
                                            align_time_step=false,
                                            verbose=false)

                    simulation.output_writers[:test] = JLD2Writer(model, (; u=model.velocities.u);
                                                                   schedule=IterationInterval(1),
                                                                   filename=filename,
                                                                   overwrite_existing=true)

                    run!(simulation)

                    field_series = FieldTimeSeries(filename * ".jld2", "u")
                    times_from_output = collect(field_series.times)
                    @test times_from_output == forcing_times

                    rm(filename * ".jld2"; force=true)
                end

                # Adaptive time-stepping
                @testset "Adaptive time stepping [$arch_type, $FT, $(clock_label)]" begin
                    grid = RectilinearGrid(arch, FT; size=(4, 4, 4), extent=(FT(1), FT(1), FT(1)))
                    forcing_times = [start_time + Dates.Hour(n) for n in 0:iterations]
                    u_forcing = FieldTimeSeries{Face, Center, Center}(grid, forcing_times)
                    for n in 1:length(forcing_times)
                        set!(u_forcing[n], (x, y, z) -> 0)
                    end

                    clock = Clock(time=start_time)
                    model = HydrostaticFreeSurfaceModel(; grid, clock, forcing=(; u=u_forcing))
                    set!(model, u=(x, y, z) -> FT(1), v=(x, y, z) -> FT(0), w=(x, y, z) -> FT(0))

                    initial_Δt = convert(FT, hour)
                    distant_stop_time = start_time + Dates.Hour(iterations + 10)
                    simulation = Simulation(model;
                                            Δt=initial_Δt,
                                            stop_iteration=iterations,
                                            stop_time=distant_stop_time,
                                            align_time_step=false,
                                            verbose=false)

                    conjure_time_step_wizard!(simulation, IterationInterval(1);
                                              cfl=0.05,
                                              max_change=1.5,
                                              min_change=0.2,
                                              max_Δt=float(initial_Δt) / 4,
                                              min_Δt=float(initial_Δt) / 10)

                    recorded_Δt = Float64[]
                    simulation.callbacks[:record_dt] = Callback(IterationInterval(1)) do sim
                        push!(recorded_Δt, float(sim.Δt))
                    end

                    run!(simulation)

                    @test length(recorded_Δt) >= iterations
                    recent_Δt = recorded_Δt[end - iterations + 1:end]
                    @test any(dt -> abs(dt - float(initial_Δt)) > 1e-6, recent_Δt)
                    @test simulation.model.clock.last_Δt != float(initial_Δt)
                    @test simulation.model.clock.time isa typeof(start_time)
                end
            end
        end
    end

    start_time = DateTime(2020, 1, 1)

    @testset "TimeInterval schedule (DateTime)" begin
        @test time_interval_schedule_checks(start_time)
    end

    @testset "SpecifiedTimes schedule (DateTime)" begin
        @test specified_times_schedule_checks(start_time)
    end

    for FT in float_types
        @testset "TimeInterval schedule [$FT]" begin
            @test numeric_time_interval_schedule_checks(FT)
        end

        @testset "SpecifiedTimes schedule [$FT]" begin
            @test numeric_specified_times_schedule_checks(FT)
        end
    end
end
