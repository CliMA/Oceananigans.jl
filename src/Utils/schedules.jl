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

#####
##### SpecifiedTimes
#####

"""
    struct SpecifiedTimes <: AbstractSchedule

Callable `TimeInterval` schedule for periodic output or diagnostic evaluation
according to `model.clock.time`.
"""
mutable struct SpecifiedTimes <: AbstractSchedule
    times :: Vector{Float64}
    previous_actuation :: Int
end

SpecifiedTimes(times::Vararg{<:Number}) = SpecifiedTimes(sort([Float64(t) for t in times]), 0)
SpecifiedTimes(times) = SpecifiedTimes(times...)

function next_appointment_time(st::SpecifiedTimes)
    if st.previous_actuation >= length(st.times)
        return Inf
    else
        return st.times[st.previous_actuation+1]
    end
end

function (st::SpecifiedTimes)(model)
    current_time = model.clock.time

    if current_time >= next_appointment_time(st)
        st.previous_actuation += 1
        return true
    end

    return false
end

align_time_step(schedule::SpecifiedTimes, clock, Δt) = min(Δt, next_appointment_time(schedule) - clock.time)

function specified_times_str(st)
    str_elems = ["$(prettytime(t)), " for t in st.times]
    str_elems = str_elems[1:end-2]
    return string("[", str_elems..., "]")
end

#####
##### Show methods
#####

show_schedule(schedule) = string(schedule)
show_schedule(schedule::IterationInterval) = string("IterationInterval(", schedule.interval, ")")
show_schedule(schedule::TimeInterval) = string("TimeInterval(", prettytime(schedule.interval), ")")
show_schedule(schedule::SpecifiedTimes) = string("SpecifiedTimes(", specified_times_str(schedule), ")")

#####
##### Any and AndSchedule
#####

struct AndSchedule{S} <: AbstractSchedule
    schedules :: S
    AndSchedule(schedules::S) where S <: Tuple = new{S}(schedules)
end

"""
    AndSchedule(child_schedule_1, child_schedule_2, other_child_schedules...)

Return a schedule that actuates when all `child_schedule`s actuate.
"""
AndSchedule(schedules...) = AndSchedule(Tuple(schedules))

# Note that multiple schedules that have a "state" (like TimeInterval and WallTimeInterval)
# could cause the logic of AndSchedule to fail, due to the short-circuiting nature of `all`.
(as::AndSchedule)(model) = all(schedule(model) for schedule in as.schedules)

struct OrSchedule{S} <: AbstractSchedule
    schedules :: S
    OrSchedule(schedules::S) where S <: Tuple = new{S}(schedules)
end

"""
    OrSchedule(child_schedule_1, child_schedule_2, other_child_schedules...)

Return a schedule that actuates when any of the `child_schedule`s actuates.
"""
OrSchedule(schedules...) = OrSchedule(Tuple(schedules))

function (as::OrSchedule)(model)
    # Ensure that all `schedules` get queried
    actuations = Tuple(schedule(model) for schedule in as.schedules)
    return any(actuations)
end

align_time_step(any_or_all_schedule::Union{OrSchedule, AndSchedule}, clock, Δt) =
    minimum(align_time_step(clock, Δt, schedule) for schedule in any_or_all_schedule.schedules)

