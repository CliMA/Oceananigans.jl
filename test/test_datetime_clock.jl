include("dependencies_for_runtests.jl")

import Dates

using Dates: DateTime
using TimesDates: TimeDate

using Oceananigans
using Oceananigans.Simulations: Simulation, run!, Callback
using Oceananigans.Units: hour, Time
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Utils: TimeInterval, SpecifiedTimes, schedule_aligned_time_step, IterationInterval
import Oceananigans: initialize!

struct DummyModel
    clock::Clock
end

function run_forcing_simulation(arch, FT, start_time; Δt, stop_time)
    grid = RectilinearGrid(arch, FT; size=(1, 1, 1), extent=(1, 1, 1))
    forcing_times = [start_time + Dates.Hour(n) for n in 0:3]
    u_forcing = FieldTimeSeries{Face, Center, Center}(grid, forcing_times)

    for n in 1:length(forcing_times)
        set!(u_forcing[n], (x, y, z) -> n - 1)
    end

    clock = Clock(time=start_time)
    model = HydrostaticFreeSurfaceModel(; grid, clock, forcing=(; u=u_forcing))
    simulation = Simulation(model; Δt, stop_time, verbose=false)

    forcing_history = FT[]
    time_history = typeof(start_time)[]

    simulation.callbacks[:capture] = Callback(IterationInterval(1)) do sim
        push!(forcing_history, u_forcing[1, 1, 1, Time(sim.model.clock.time)])
        push!(time_history, sim.model.clock.time)
    end

    run!(simulation)

    return forcing_history, time_history
end

function time_interval_schedule_checks(start_time)
    schedule = TimeInterval(Dates.Hour(1))
    clock = Clock(time=start_time)
    model = DummyModel(clock)

    initialize!(schedule, model)
    @test !schedule(model)

    # Alignment trims the step to the next hourly mark
    Δt = hour + 1
    @test schedule_aligned_time_step(schedule, clock, Δt) == hour

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

    Δt = 3
    @test schedule_aligned_time_step(schedule, clock, Δt) == 2

    tick!(clock, 2)
    @test schedule(model)
    @test !schedule(model)

    tick!(clock, 1)
    aligned = schedule_aligned_time_step(schedule, clock, 5)
    @test aligned == 1

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

    Δt = 5
    @test schedule_aligned_time_step(schedule, clock, Δt) == 1

    tick!(clock, 1)
    @test schedule(model)
    @test !schedule(model)

    tick!(clock, 1)
    aligned = schedule_aligned_time_step(schedule, clock, 2)
    @test aligned == 1

    tick!(clock, aligned)
    @test schedule(model)
    @test !schedule(model)

    @test schedule_aligned_time_step(schedule, clock, 1) == 1

    return true
end

@testset "DateTime clocks" begin
    @info "Testing DateTime clock behavior..."

    test_date_types = [DateTime, TimeDate]

    for arch in archs, FT in float_types
        arch_type = typeof(arch)
        for TimeType in test_date_types
            start_time = TimeType(2020, 1, 1)
            stop_time = start_time + Dates.Hour(3)

            test_Δt = [1200, 3600, Dates.Minute(20), Dates.Hour(1)]

            for Δt in test_Δt
                forcing_history, time_history = run_forcing_simulation(arch, FT, start_time; Δt, stop_time)

                DT = typeof(Δt)
                @testset "Hydrostatic $TimeType forcing [$arch_type, $FT, $DT]" begin
                    Nt = 3 * 3600 / Oceananigans.Utils.period_to_seconds(Δt)
                    expected_times = [start_time + Δt for n in 0:Nt]
                    expected_forcing = [n - 1 for n in 0:Nt]
                    @test time_history == expected_times
                    @test forcing_history == expected_forcing
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
