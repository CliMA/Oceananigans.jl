"""
    AbstractTrigger

Supertype for objects that trigger `OutputWriter`s and `Diagnostics`.
Triggers must define a function `Trigger(model)` that returns true or
false.
"""
abstract type AbstractTrigger end

initialize!(abstract_trigger) = nothing # fallback

#####
##### TimeInterval
#####

"""
    struct TimeInterval <: AbstractTrigger

Callable `TimeInterval` trigger for periodic output or diagnostic evaluation
according to `model.clock.time`.
"""
mutable struct TimeInterval <: AbstractTrigger
    time_interval :: Float64
    previous_triggering_time :: Float64
end

"""
    TimeInterval(time_interval)

Returns a callable `TimeInterval` that triggers periodic output or diagnostic evaluation
on a `time_interval` of simulation time, as kept by `model.clock`.
"""
TimeInterval(time_interval) = TimeInterval(Float64(time_interval), 0.0)

function (trigger::TimeInterval)(model)
    time = model.clock.time

    if time >= trigger.previous_triggering_time + trigger.time_interval
        # Shave the overshoot off previous_triggering_time to prevent overshoot
        # from accumulating
        trigger.previous_triggering_time = time - rem(time, trigger.time_interval)
        return true
    else
        return false
    end

end

#####
##### IterationInterval
#####

"""
    IterationInterval(iteration_interval)

Returns a callable IterationInterval that triggers periodic output or diagnostic evaluation
over `iteration_interval`, and therefore when `model.clock.iteration % trigger.iteration_interval == 0`.
"""
struct IterationInterval <: AbstractTrigger
    iteration_interval :: Int
end

(trigger::IterationInterval)(model) = model.clock.iteration % trigger.iteration_interval == 0

"""
    mutable struct WallTimeInterval <: AbstractTrigger

"""
mutable struct WallTimeInterval <: AbstractTrigger
    time_interval :: Float64
    previous_triggering_time :: Float64
end

"""
    WallTimeInterval(time_interval)

Returns a callable WallTimeInterval that triggers periodic output or diagnostic evaluation
on a `time_interval` of "wall time" while a simulation runs. The "wall time"
is the actual real world time, as kept by an actual or hypothetical clock hanging
on your wall.
"""
WallTimeInterval(time_interval) = WallTimeInterval(Float64(time_interval), time_ns() * 1e-9)

initialize!(trigger::WallTimeInterval) = trigger.previous_triggering_time = time_ns() * 1e-9

function (trigger::TimeInterval)(model)
    wall_time = time_ns() * 1e-9

    if wall_time >= trigger.previous_triggering_time + trigger.time_interval
        # Shave the overshoot off previous_triggering_time to prevent overshoot
        # from accumulating
        trigger.previous_triggering_time = wall_time - rem(wall_time, trigger.time_interval)
        return true
    else
        return false
    end

end
