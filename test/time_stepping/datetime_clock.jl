include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

import Dates

using Dates: DateTime
using TimesDates: TimeDate

using Oceananigans
using Oceananigans.Simulations: Simulation, run!, Callback, add_callback!
using Oceananigans.Units: hour, Time
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Utils: TimeInterval, SpecifiedTimes, schedule_aligned_time_step, IterationInterval
using Oceananigans.OutputWriters: AveragedTimeInterval
using Oceananigans.OutputReaders: Cyclical
import Oceananigans: initialize!

struct DummyModel
    clock::Clock
end

function run_forcing_simulation(arch, FT, start_time; Δt, stop_time, free_surface)
    grid = RectilinearGrid(arch, FT; size=(1, 1, 1), extent=(1, 1, 1))
    forcing_times = [start_time + Dates.Hour(n) for n in 0:3]
    u_forcing = FieldTimeSeries{Face, Center, Center}(grid, forcing_times)

    for n in 1:length(forcing_times)
        set!(u_forcing[n], (x, y, z) -> n - 1)
    end

    clock = Clock(time=start_time)
    model = HydrostaticFreeSurfaceModel(grid; clock, free_surface, forcing=(; u=u_forcing))
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

function averaged_time_interval_schedule_checks(start_time)
    schedule = AveragedTimeInterval(Dates.Hour(2), window=Dates.Hour(1))
    @test schedule.interval isa Dates.Hour
    @test schedule.window  isa Dates.Hour
    @test schedule.first_actuation_time isa DateTime

    clock = Clock(time=start_time)
    model = DummyModel(clock)

    initialize!(schedule, model)
    @test schedule.first_actuation_time == start_time
    @test !schedule(model)

    tick!(clock, hour + 60)
    @test schedule(model)

    @test_throws ArgumentError initialize!(AveragedTimeInterval(1.0), DummyModel(Clock(time=start_time)))

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

function numeric_averaged_time_interval_schedule_checks(FT)
    schedule = AveragedTimeInterval(FT(2), window=FT(1))
    @test schedule.interval isa FT
    @test schedule.window  isa FT
    @test schedule.first_actuation_time isa FT

    clock = Clock(time=zero(FT))
    model = DummyModel(clock)

    initialize!(schedule, model)
    @test schedule.first_actuation_time == zero(FT)
    @test !schedule(model)

    tick!(clock, FT(1.5))
    @test schedule(model)

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

function calendar_cyclical_checks()
    grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))

    # A monthly climatology stamped mid-month in 2018, valued by month
    months = [DateTime(2018, m, 15) for m in 1:12]
    climatology = FieldTimeSeries{Center, Center, Center}(grid, months; time_indexing=Cyclical(Dates.Year(1)))
    for n in 1:12
        set!(climatology[n], n)
    end

    c(t) = climatology[1, 1, 1, Time(t)]
    @test c(DateTime(1958, 3, 15)) == 3
    @test c(DateTime(2043, 7, 15)) == 7
    @test c(DateTime(1993, 12, 31)) ≈ 12 + 16 / 31 * (1 - 12) # wraps from December to January
    @test c(DateTime(1994, 1, 1))   ≈ 12 + 17 / 31 * (1 - 12)

    # A 3-hourly repeat year in 1990 (not a leap year), valued by hours since its start
    hours = DateTime(1990) : Dates.Hour(3) : DateTime(1990, 12, 31, 21)
    repeat_year = FieldTimeSeries{Center, Center, Center}(grid, collect(hours); time_indexing=Cyclical(Dates.Year(1)))
    for n in eachindex(hours)
        set!(repeat_year[n], 3 * (n - 1))
    end

    r(t) = repeat_year[1, 1, 1, Time(t)]
    @test r(DateTime(2044, 2, 29, 12)) == r(DateTime(2044, 2, 28, 12)) # 29 February reads the 28th
    @test r(DateTime(2044, 3, 1, 12)) == 24 * (31 + 28) + 12

    # A cycle of several years
    years = [DateTime(y, 7, 1) for y in 1958:1960]
    cycle = FieldTimeSeries{Center, Center, Center}(grid, years; time_indexing=Cyclical(Dates.Year(3)))
    for n in 1:3
        set!(cycle[n], n)
    end

    @test cycle[1, 1, 1, Time(DateTime(1961, 7, 1))] == 1
    @test cycle[1, 1, 1, Time(DateTime(1957, 7, 1))] == 3

    return true
end

function calendar_cyclical_forcing(arch)
    grid = RectilinearGrid(arch; size=(1, 1, 1), extent=(1, 1, 1))
    months = [DateTime(2018, m, 15) for m in 1:12]
    climatology = FieldTimeSeries{Center, Center, Center}(grid, months; time_indexing=Cyclical(Dates.Year(1)))
    for n in 1:12
        set!(climatology[n], n)
    end

    forcing = Forcing((i, j, k, grid, clock, fields, p) -> @inbounds(p[i, j, k, Time(clock.time)]),
                      discrete_form=true, parameters=climatology)

    clock = Clock(time=DateTime(1993, 3, 15))
    model = HydrostaticFreeSurfaceModel(grid; clock, tracers=:c, buoyancy=nothing, forcing=(; c=forcing))
    time_step!(model, 1)

    return model.tracers.c[1, 1, 1]
end

@testset "DateTime clocks" begin
    test_date_types = [DateTime, TimeDate]

    for arch in archs, FT in float_types
        arch_type = typeof(arch)
        for TimeType in test_date_types
            start_time = TimeType(2020, 1, 1)
            stop_time = start_time + Dates.Hour(3)

            test_Δt = [1200, 3600, Dates.Minute(20), Dates.Hour(1)]

            # The split-explicit substeps advance a substep clock and load the forcing window past tⁿ⁺¹
            free_surfaces = (ImplicitFreeSurface(), SplitExplicitFreeSurface(substeps=4))

            @testset "Hydrostatic $(TimeType) forcing [$(arch_type), $(FT), $(typeof(Δt)), $(nameof(typeof(free_surface)))]" for Δt in test_Δt, free_surface in free_surfaces
                forcing_history, time_history = run_forcing_simulation(arch, FT, start_time; Δt, stop_time, free_surface)

                Nt = 3 * 3600 / Oceananigans.Utils.period_to_seconds(Δt)
                Nt = Int(Nt)

                Δt_period = Dates.Second(Δt)
                expected_times = [start_time + n * Δt_period for n in 0:Nt]

                dF = 3 / Nt # determined by run_forcing_simulation
                expected_forcing = [0 + dF * n for n in 0:Nt]

                @test time_history == expected_times
                @test forcing_history ≈ expected_forcing
            end
        end
    end

    @testset "Cyclical FieldTimeSeries over calendar years" begin
        @test calendar_cyclical_checks()

        for arch in archs
            @test calendar_cyclical_forcing(arch) == 3 # March, after a step of 1 second
        end
    end

    @testset "Numeric schedules on a calendar clock" begin
        grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
        model = HydrostaticFreeSurfaceModel(grid; clock=Clock(time=DateTime(2020, 1, 1)))
        simulation = Simulation(model; Δt=60, stop_iteration=1, verbose=false)
        add_callback!(simulation, sim -> nothing, TimeInterval(hour))
        @test_throws ArgumentError run!(simulation)
    end

    @testset "Numeric stop time on a calendar clock" begin
        grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
        model = HydrostaticFreeSurfaceModel(grid; clock=Clock(time=DateTime(2020, 1, 1)))
        @test_throws ArgumentError Simulation(model; Δt=60, stop_time=2hour)
    end

    @testset "Simulation without a stop time [$(TimeType)]" for TimeType in test_date_types
        grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
        model = HydrostaticFreeSurfaceModel(grid; clock=Clock(time=TimeType(2020, 1, 1)))
        simulation = Simulation(model; Δt=60, stop_iteration=3, verbose=false)
        run!(simulation)
        @test model.clock.time == TimeType(2020, 1, 1, 0, 3)
    end

    start_time = DateTime(2020, 1, 1)

    @testset "TimeInterval schedule (DateTime)" begin
        @test time_interval_schedule_checks(start_time)
    end

    @testset "AveragedTimeInterval schedule (DateTime)" begin
        @test averaged_time_interval_schedule_checks(start_time)
    end

    @testset "SpecifiedTimes schedule (DateTime)" begin
        @test specified_times_schedule_checks(start_time)
    end

    for FT in float_types
        @testset "TimeInterval schedule [$FT]" begin
            @test numeric_time_interval_schedule_checks(FT)
        end

        @testset "AveragedTimeInterval schedule [$FT]" begin
            @test numeric_averaged_time_interval_schedule_checks(FT)
        end

        @testset "SpecifiedTimes schedule [$FT]" begin
            @test numeric_specified_times_schedule_checks(FT)
        end
    end
end
