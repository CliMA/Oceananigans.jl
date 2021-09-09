"""
    AbstractSchedule

Supertype for objects that schedule `OutputWriter`s and `Diagnostics`.
Schedules must define the functor `Schedule(model)` that returns true or
false.
"""
abstract type AbstractSchedule end

initialize_schedule!(schedule) = nothing # fallback

# Default behavior is no alignment.
pruned_time_step(schedule, clock, Δt) = Δt

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

function pruned_time_step(schedule::TimeInterval, clock, Δt)
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

"""
    mutable struct WallTimeInterval <: AbstractSchedule

"""
mutable struct WallTimeInterval <: AbstractSchedule
    interval :: Float64
    previous_actuation_time :: Float64
end

"""
    WallTimeInterval(interval)

Returns a callable WallTimeInterval that schedules periodic output or diagnostic evaluation
on a `interval` of "wall time" while a simulation runs. The "wall time"
is the actual real world time, as kept by an actual or hypothetical clock hanging
on your wall.
"""
WallTimeInterval(interval) = WallTimeInterval(Float64(interval), time_ns() * 1e-9)

initialize_schedule!(schedule::WallTimeInterval) = schedule.previous_actuation_time = time_ns() * 1e-9

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
