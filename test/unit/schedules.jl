include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval, SpecifiedTimes, ConsecutiveIterations, TimeOffset
using Oceananigans.Utils: schedule_aligned_time_step, next_actuation_time
using Oceananigans.TimeSteppers: Clock
using Oceananigans: initialize!, prognostic_state, restore_prognostic_state!
using Dates: Second, Minute

@testset "Schedules" begin
    @info "Testing schedules..."

    # Some fake models
    fake_model_at_iter_0 = (; clock=Clock(time=0.0, iteration=0))
    fake_model_at_iter_2 = (; clock=Clock(time=0.0, iteration=2))
    fake_model_at_iter_3 = (; clock=Clock(time=1.0, iteration=3))
    fake_model_at_iter_4 = (; clock=Clock(time=2.1, iteration=4))
    fake_model_at_iter_5 = (; clock=Clock(time=2.0, iteration=5))

    fake_model_at_time_2 = (; clock=Clock(time=2.0, iteration=3))
    fake_model_at_time_3 = (; clock=Clock(time=3.0, iteration=3))
    fake_model_at_time_4 = (; clock=Clock(time=4.0, iteration=1))
    fake_model_at_time_5 = (; clock=Clock(time=5.0, iteration=1))

    # TimeInterval
    ti = TimeInterval(2)
    initialize!(ti, fake_model_at_iter_0)

    @test ti.actuations == 0
    @test ti.interval == 2.0
    @test ti(fake_model_at_time_2)
    @test !(ti(fake_model_at_time_3))
    @test initialize!(ti, fake_model_at_iter_0)

    # Catchup behavior
    ti_catchup = TimeInterval(2)
    initialize!(ti_catchup, fake_model_at_iter_0)
    far_future_model = (; clock=Clock(time=100.0, iteration=1000))

    @test ti_catchup(far_future_model)
    @test !(ti_catchup(far_future_model))
    @test next_actuation_time(ti_catchup) > 100.0

    # Normal one-firing-per-crossing is preserved
    ti_normal = TimeInterval(2)
    initialize!(ti_normal, fake_model_at_iter_0)
    @test ti_normal((; clock=Clock(time=2.5, iteration=1)))
    @test !(ti_normal((; clock=Clock(time=2.5, iteration=1))))
    @test ti_normal.actuations == 1

    # Array-interval TimeInterval (used by `TimeInterval(::AveragedSpecifiedTimes)`):
    ti_array = TimeInterval([1.0])
    @test ti_array((; clock=Clock(time=1.0, iteration=10)))
    @test ti_array.actuations == 1
    @test next_actuation_time(ti_array) === Inf
    @test !(ti_array((; clock=Clock(time=2.0, iteration=20))))
    @test schedule_aligned_time_step(ti_array, Clock(time=2.0, iteration=20), 0.1) == 0.1

    ti_array_multi = TimeInterval([1.0, 2.0, 3.0])
    @test ti_array_multi((; clock=Clock(time=5.0, iteration=50)))
    @test ti_array_multi.actuations == 3
    @test next_actuation_time(ti_array_multi) === Inf
    @test !(ti_array_multi((; clock=Clock(time=6.0, iteration=60))))

    # Restore across pickup with the SAME interval: phase preserved.
    ti_old = TimeInterval(2)
    ti_old.first_actuation_time = 0.0
    ti_old.actuations = 42
    state = prognostic_state(ti_old)
    @test haskey(state, :interval)

    ti_same = TimeInterval(2)
    restore_prognostic_state!(ti_same, state)
    @test ti_same.first_actuation_time == 0.0
    @test ti_same.actuations == 42
    @test next_actuation_time(ti_same) == 86.0

    # Restore across pickup with a different interval
    ti_changed = TimeInterval(10)
    restore_prognostic_state!(ti_changed, state)
    @test ti_changed.first_actuation_time == 0.0
    @test ti_changed.actuations == 0
    @test ti_changed.interval == 10.0

    # And that schedule, when called with a clock far in the future, fires
    # once and aligns subsequent fires to the new interval's phase grid.
    far_clock = (; clock=Clock(time=95.0, iteration=42))
    @test ti_changed(far_clock)
    @test !(ti_changed(far_clock))
    @test next_actuation_time(ti_changed) == 100.0

    # IterationInterval
    ii = IterationInterval(3)

    @test !(ii(fake_model_at_iter_5))
    @test ii(fake_model_at_iter_3)
    @test initialize!(ii, fake_model_at_iter_0)

    old_time_interval_state = (first_actuation_time = 0.0, actuations = 7, interval = 2.0)
    restore_prognostic_state!(ii, old_time_interval_state)
    @test ii.interval == 3
    @test ii.offset == 0
    @test ii(fake_model_at_iter_3)

    # OrSchedule
    ti_and_ii = AndSchedule(TimeInterval(2), IterationInterval(3))
    @test ti_and_ii(fake_model_at_time_2)
    @test !(ti_and_ii(fake_model_at_time_4))
    @test !(ti_and_ii(fake_model_at_iter_3))
    @test !(ti_and_ii(fake_model_at_iter_5))
    @test !(ti_and_ii(fake_model_at_time_3))

    ti_or_ii = OrSchedule(TimeInterval(2), IterationInterval(3))
    @test ti_or_ii(fake_model_at_iter_3)
    @test ti_or_ii(fake_model_at_iter_5) # triggers TimeInterval but not IterationInterval
    @test ti_or_ii(fake_model_at_time_3) # triggers IterationInterval but not TimeInterval
    @test ti_or_ii(fake_model_at_time_4) # triggers TimeInterval but not IterationInterval
    @test !(ti_or_ii(fake_model_at_time_5))

    ii_plus_one = ConsecutiveIterations(IterationInterval(3))
    @test !(ii_plus_one(fake_model_at_iter_2))
    @test ii_plus_one(fake_model_at_iter_3)
    @test ii_plus_one(fake_model_at_iter_4)
    @test !(ti_or_ii(fake_model_at_iter_5))

    ti_plus_one = ConsecutiveIterations(TimeInterval(2))
    @test ti_plus_one(fake_model_at_time_2) # and iter 3
    @test ti_plus_one(fake_model_at_iter_4)
    @test !(ti_plus_one(fake_model_at_iter_5))

    # TimeOffset with a positive offset: actuates with the parent and again `offset` later
    ii_then_later = TimeOffset(IterationInterval(3), 0.5)
    @test !(ii_then_later(fake_model_at_iter_2))
    @test ii_then_later(fake_model_at_iter_3) # parent actuates at t = 1
    @test 0.5 ≈ schedule_aligned_time_step(ii_then_later, fake_model_at_iter_3.clock, Inf)
    @test !(ii_then_later((; clock=Clock(time=1.2, iteration=4))))
    @test ii_then_later((; clock=Clock(time=1.5, iteration=5))) # offset actuation at t = 1.5
    @test !(ii_then_later((; clock=Clock(time=1.7, iteration=5))))
    @test Inf == schedule_aligned_time_step(ii_then_later, Clock(time=1.7, iteration=5), Inf)

    # TimeOffset with a negative offset: actuates `|offset|` before the parent, and with the parent
    ti_and_before = TimeOffset(TimeInterval(2), -0.5)
    @test initialize!(ti_and_before, fake_model_at_iter_0)
    @test !(ti_and_before((; clock=Clock(time=1.0, iteration=1))))
    @test 0.5 ≈ schedule_aligned_time_step(ti_and_before, Clock(time=1.0, iteration=1), Inf)
    @test ti_and_before((; clock=Clock(time=1.5, iteration=2))) # offset actuation at t = 1.5
    @test !(ti_and_before((; clock=Clock(time=1.7, iteration=3))))
    @test 0.3 ≈ schedule_aligned_time_step(ti_and_before, Clock(time=1.7, iteration=3), Inf) # aligned with the parent
    @test ti_and_before(fake_model_at_time_2) # parent actuation at t = 2
    @test !(ti_and_before(fake_model_at_time_3))
    @test 0.5 ≈ schedule_aligned_time_step(ti_and_before, Clock(time=3.0, iteration=3), Inf)
    @test ti_and_before((; clock=Clock(time=3.5, iteration=4))) # offset actuation at t = 3.5

    restored_ti_and_before = TimeOffset(TimeInterval(2), -0.5)
    restore_prognostic_state!(restored_ti_and_before, prognostic_state(ti_and_before))
    @test restored_ti_and_before.offset_actuated == ti_and_before.offset_actuated
    @test restored_ti_and_before.parent_actuation_time == ti_and_before.parent_actuation_time
    @test restored_ti_and_before.parent.actuations == ti_and_before.parent.actuations
    @test isnothing(restore_prognostic_state!(restored_ti_and_before, nothing))
    @test summary(ti_and_before) == "TimeOffset(TimeInterval(2 seconds), -500 ms)"

    @test_throws ArgumentError TimeOffset(IterationInterval(3), -0.5)
    @test_throws ArgumentError TimeOffset(TimeInterval(2), -2)
    @test_throws ArgumentError TimeOffset(TimeInterval(2), 3)

    # A positive offset is pending only after the parent actuates, and a new parent actuation resets it
    st_then_later = TimeOffset(SpecifiedTimes(1.0, 1.2), 0.5)
    @test !(initialize!(st_then_later, fake_model_at_iter_0))
    @test !(st_then_later((; clock=Clock(time=0.5, iteration=1))))
    @test 0.5 ≈ schedule_aligned_time_step(st_then_later, Clock(time=0.5, iteration=1), Inf)
    @test st_then_later((; clock=Clock(time=1.0, iteration=2))) # parent actuation at t = 1
    @test 0.2 ≈ schedule_aligned_time_step(st_then_later, Clock(time=1.0, iteration=2), Inf)
    @test st_then_later((; clock=Clock(time=1.2, iteration=3))) # parent actuation at t = 1.2 moves the offset to t = 1.7
    @test !(st_then_later((; clock=Clock(time=1.5, iteration=4))))
    @test 0.2 ≈ schedule_aligned_time_step(st_then_later, Clock(time=1.5, iteration=4), Inf)
    @test st_then_later((; clock=Clock(time=1.7, iteration=5))) # offset actuation at t = 1.7
    @test !(st_then_later((; clock=Clock(time=2.0, iteration=6))))
    @test Inf == schedule_aligned_time_step(st_then_later, Clock(time=2.0, iteration=6), Inf)

    # A negative offset stops actuating once the parent runs out of actuation times
    st_and_before = TimeOffset(SpecifiedTimes(1.0, 2.0), -0.5)
    @test !(initialize!(st_and_before, fake_model_at_iter_0))
    @test st_and_before((; clock=Clock(time=0.5, iteration=1)))
    @test st_and_before((; clock=Clock(time=1.0, iteration=2)))
    @test st_and_before((; clock=Clock(time=1.5, iteration=3)))
    @test st_and_before((; clock=Clock(time=2.0, iteration=4)))
    @test !(st_and_before((; clock=Clock(time=2.5, iteration=5))))
    @test Inf == schedule_aligned_time_step(st_and_before, Clock(time=2.5, iteration=5), Inf)

    # Function parents work with positive offsets
    at_one(model) = model.clock.time == 1
    func_then_later = TimeOffset(at_one, 0.5)
    @test !(initialize!(func_then_later, fake_model_at_iter_0))
    @test !(func_then_later((; clock=Clock(time=0.5, iteration=1))))
    @test func_then_later((; clock=Clock(time=1.0, iteration=2)))
    @test !(func_then_later((; clock=Clock(time=1.2, iteration=3))))
    @test 0.3 ≈ schedule_aligned_time_step(func_then_later, Clock(time=1.2, iteration=3), Inf)
    @test func_then_later((; clock=Clock(time=1.5, iteration=4)))
    @test !(func_then_later((; clock=Clock(time=1.7, iteration=5))))

    # DateTime clocks with Dates.Period offsets
    start_time = DateTime(2025, 1, 1)
    datetime_model(seconds) = (; clock=Clock(time=start_time + Second(seconds), iteration=0))
    minute_and_before = TimeOffset(TimeInterval(Minute(1)), Second(-10))
    @test initialize!(minute_and_before, datetime_model(0))
    @test !(minute_and_before(datetime_model(30)))
    @test 20 ≈ schedule_aligned_time_step(minute_and_before, datetime_model(30).clock, Inf)
    @test minute_and_before(datetime_model(50)) # offset actuation
    @test 10 ≈ schedule_aligned_time_step(minute_and_before, datetime_model(50).clock, Inf)
    @test minute_and_before(datetime_model(60)) # parent actuation
    @test !(minute_and_before(datetime_model(90)))
    @test minute_and_before(datetime_model(110))
    @test summary(minute_and_before) == "TimeOffset(TimeInterval(1 minute), -10 seconds)"
    @test_throws ArgumentError initialize!(TimeOffset(TimeInterval(Minute(1)), -10), datetime_model(0))

    # WallTimeInterval
    wti = WallTimeInterval(1e-9)

    @test wti.interval == 1e-9
    @test wti(nothing)

    # SpecifiedTimes
    st = st_list = SpecifiedTimes(2, 5, 6)
    st_vector = SpecifiedTimes([2, 5, 6])
    @test st_list.times == st_vector.times
    @test st.times == [2.0, 5.0, 6.0]
    @test !(initialize!(st, fake_model_at_iter_0))

    # Times are sorted
    st = SpecifiedTimes(5, 2, 6)
    @test st.times == [2.0, 5.0, 6.0]

    @test st(fake_model_at_time_2)

    @test !(st(fake_model_at_time_4))
    @test st(fake_model_at_time_5)

    # Specified times includes iteration 0
    st = SpecifiedTimes(0, 2, 4)
    @test initialize!(st, fake_model_at_iter_0)

    fake_clock = (; time=2.1)
    st = SpecifiedTimes(2.5)
    @test 0.4 ≈ schedule_aligned_time_step(st, fake_clock, Inf)
end

@testset "TimeOffset time step alignment" begin
    @info "Testing TimeOffset time step alignment..."

    grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))

    for (offset, expected_times) in [(-0.2, [0.0, 0.8, 1.0, 1.8, 2.0]),
                                     (+0.3, [0.0, 0.3, 1.0, 1.3, 2.0])]
        model = NonhydrostaticModel(grid)
        simulation = Simulation(model; Δt=0.3, stop_time=2, verbose=false)

        actuation_times = Float64[]
        record_time(sim) = push!(actuation_times, time(sim))
        add_callback!(simulation, record_time, TimeOffset(TimeInterval(1), offset))

        dir = mktempdir()
        filename = "offset_actuation.jld2"
        simulation.output_writers[:u] = JLD2Writer(model, (; u=model.velocities.u); dir, filename,
                                                   schedule=TimeOffset(TimeInterval(1), offset))
        run!(simulation)

        @test actuation_times ≈ expected_times
        @test FieldTimeSeries(joinpath(dir, filename), "u").times ≈ expected_times
    end
end
