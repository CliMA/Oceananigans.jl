include("dependencies_for_runtests.jl")

import Dates

using Oceananigans
using Oceananigans.Units: hour, Time
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Utils: TimeInterval, SpecifiedTimes, schedule_aligned_time_step
import Oceananigans: initialize!

struct DummyModel
    clock::Clock
end

function time_step_hydrostatic_with_datetime_field_time_series_forcing(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    start_time = Dates.DateTime(2020, 1, 1)
    times = start_time:Dates.Hour(1):start_time + Dates.Hour(3)

    u_forcing = FieldTimeSeries{Face, Center, Center}(grid, times)

    for (n, _) in enumerate(times)
        set!(u_forcing[n], (x, y, z) -> Float64(n - 1))
    end

    clock = Clock(time=start_time)
    model = HydrostaticFreeSurfaceModel(; grid, clock, forcing=(; u=u_forcing))
    forcing_history = Float64[]
    time_history = Dates.DateTime[]

    push!(forcing_history, u_forcing[1, 1, 1, Time(model.clock.time)])
    push!(time_history, model.clock.time)

    time_step!(model, hour; euler=true)
    push!(forcing_history, u_forcing[1, 1, 1, Time(model.clock.time)])
    push!(time_history, model.clock.time)

    for _ in 1:2
        time_step!(model, hour)
        push!(forcing_history, u_forcing[1, 1, 1, Time(model.clock.time)])
        push!(time_history, model.clock.time)
    end

    expected_times = [start_time + Dates.Hour(n) for n in 0:3]
    expected_forcing = Float64.(0:3)

    @test time_history == expected_times
    @test forcing_history == expected_forcing

    return true
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

@testset "DateTime clocks" begin
    @info "Testing DateTime clock behavior..."

    for arch in archs
        @testset "Hydrostatic DateTime forcing [$(typeof(arch))]" begin
            @test time_step_hydrostatic_with_datetime_field_time_series_forcing(arch)
        end
    end

    start_time = Dates.DateTime(2020, 1, 1)

    @testset "TimeInterval schedule" begin
        @test time_interval_schedule_checks(start_time)
    end

    @testset "SpecifiedTimes schedule" begin
        @test specified_times_schedule_checks(start_time)
    end
end
