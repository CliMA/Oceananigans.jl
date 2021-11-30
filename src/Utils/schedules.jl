"""
    AbstractSchedule

Supertype for objects that schedule `OutputWriter`s and `Diagnostics`.
Schedule must define the functor `Schedule(model)` that returns true or
false.
"""
abstract type AbstractSchedule end

# Default behavior is no alignment.
aligned_time_step(schedule, clock, Δt) = Δt

#####
##### TimeInterval
#####

"""
    struct TimeInterval <: AbstractSchedule

Callable `TimeInterval` schedule for periodic output or diagnostic evaluation
according to `model.clock.time`.
"""
mutable struct TimeInterval <: AbstractSchedule
    interval :: Float64
    previous_actuation_time :: Float64
end

"""
    TimeInterval(interval)

Returns a callable `TimeInterval` that schedules periodic output or diagnostic evaluation
on a `interval` of simulation time, as kept by `model.clock`.
"""
TimeInterval(interval) = TimeInterval(Float64(interval), 0.0)

function (schedule::TimeInterval)(model)
    time = model.clock.time

    if time == schedule.previous_actuation_time + schedule.interval
        schedule.previous_actuation_time = time
        return true
    elseif time > schedule.previous_actuation_time + schedule.interval
        # Shave overshoot off previous_actuation_time to prevent overshoot from accumulating
        schedule.previous_actuation_time = time - rem(time, schedule.interval)
        return true
    else
        return false
    end
end

function aligned_time_step(schedule::TimeInterval, clock, Δt)
    next_actuation_time = schedule.previous_actuation_time + schedule.interval
    return min(Δt, next_actuation_time - clock.time)
end

#####
##### IterationInterval
#####

"""
    IterationInterval(interval)

Returns a callable IterationInterval that schedules periodic output or diagnostic evaluation
over `interval`, and therefore when `model.clock.iteration % schedule.interval == 0`.
"""
struct IterationInterval <: AbstractSchedule
    interval :: Int
end

(schedule::IterationInterval)(model) = model.clock.iteration % schedule.interval == 0

#####
##### WallTimeInterval
#####

mutable struct WallTimeInterval <: AbstractSchedule
    interval :: Float64
    previous_actuation_time :: Float64
end

"""
    WallTimeInterval(interval; start_time = time_ns() * 1e-9)

Returns a callable WallTimeInterval that schedules periodic output or diagnostic evaluation
on a `interval` of "wall time" while a simulation runs, in units of seconds.

The "wall time" is the actual real world time in seconds, as kept by an actual
or hypothetical clock hanging on your wall.

The keyword argument `start_time` can be used to specify a starting wall time
other than the moment `WallTimeInterval` is constructed.
"""
WallTimeInterval(interval; start_time = time_ns() * 1e-9) = WallTimeInterval(Float64(interval), Float64(start_time))

function (schedule::WallTimeInterval)(model)
    wall_time = time_ns() * 1e-9

    if wall_time >= schedule.previous_actuation_time + schedule.interval
        # Shave overshoot off previous_actuation_time to prevent overshoot from accumulating
        schedule.previous_actuation_time = wall_time - rem(wall_time, schedule.interval)
        return true
    else
        return false
    end
end

show_schedule(schedule) = string(schedule)
show_schedule(schedule::IterationInterval) = string("IterationInterval(", schedule.interval, ")")
show_schedule(schedule::TimeInterval) = string("TimeInterval(", prettytime(schedule.interval), ")")

#####
##### All and any schedules
#####

struct AllSchedule{S} <: AbstractSchedule
    schedules :: S
    AllSchedule(schedules::S) where S <: Tuple = new{S}(schedules)
end

"""
    AllSchedule(child_schedule_1, child_schedule_2, other_child_schedules...)

Return a schedule that actuates when all `child_schedule`s actuate.
"""
AllSchedule(schedules...) = AllSchedule(Tuple(schedules))

(as::AllSchedule)(model) = all(schedule(model) for schedule in as.schedules)

struct AnySchedule{S} <: AbstractSchedule
    schedules :: S
    AnySchedule(schedules::S) where S <: Tuple = new{S}(schedules)
end

"""
    AnySchedule(child_schedule_1, child_schedule_2, other_child_schedules...)

Return a schedule that actuates when any of the `child_schedule`s actuates.
"""
AnySchedule(schedules...) = AnySchedule(Tuple(schedules))

(as::AnySchedule)(model) = any(schedule(model) for schedule in as.schedules)

