include("dependencies_for_runtests.jl")

using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval, SpecifiedTimes, ConsecutiveIterations
using Oceananigans.Utils: schedule_aligned_time_step, next_actuation_time
using Oceananigans.TimeSteppers: Clock
using Oceananigans: initialize!, prognostic_state, restore_prognostic_state!

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
