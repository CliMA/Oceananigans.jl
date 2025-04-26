import Oceananigans: initialize!

"""
    AbstractSchedule

Supertype for objects that schedule `OutputWriter`s and `Diagnostics`.
Schedule must define the functor `Schedule(model)` that returns true or
false.
"""
abstract type AbstractSchedule end

# Default behavior is no alignment.
schedule_aligned_time_step(schedule, clock, Δt) = Δt

# Fallback initialization for schedule: call the schedule,
# then return `true`, indicating that the schedule "actuates" at
# initial call.
function initialize!(schedule::AbstractSchedule, model)
    schedule(model)

    # the default behavior `return true` dictates that by default,
    # schedules actuate at the initial call.
    return true
end

#####
##### TimeInterval
#####

"""
    struct TimeInterval <: AbstractSchedule

Callable `TimeInterval` schedule for periodic output or diagnostic evaluation
according to `model.clock.time`.
"""
mutable struct TimeInterval{FT} <: AbstractSchedule
    interval :: FT
    first_actuation_time :: FT
    actuations :: Int
end

"""
    TimeInterval(interval)

Return a callable `TimeInterval` that schedules periodic output or diagnostic evaluation
on a `interval` of simulation time, as kept by `model.clock`.
"""
function TimeInterval(interval)
    FT = Oceananigans.defaults.FloatType
    interval = convert(FT, interval)
    return TimeInterval(interval, zero(FT), 0)
end

function initialize!(schedule::TimeInterval, first_actuation_time::Number)
    schedule.first_actuation_time = first_actuation_time
    schedule.actuations = 0
    return true
end

initialize!(schedule::TimeInterval, model) = initialize!(schedule, model.clock.time)

function next_actuation_time(schedule::TimeInterval)
    t₀ = schedule.first_actuation_time
    N = schedule.actuations
    T = schedule.interval
    return t₀ + (N + 1) * T
end

function (schedule::TimeInterval)(model)
    t = model.clock.time
    t★ = next_actuation_time(schedule)

    if t >= t★
        if schedule.actuations < typemax(Int)
            schedule.actuations += 1
        else # re-initialize the schedule to t★
            initialize!(schedule, t★)
        end
        return true
    else
        return false
    end
end

function schedule_aligned_time_step(schedule::TimeInterval, clock, Δt)
    t★ = next_actuation_time(schedule)
    t = clock.time
    return min(Δt, t★ - t)
end

#####
##### IterationInterval
#####

struct IterationInterval <: AbstractSchedule
    interval :: Int
    offset :: Int
end

"""
    IterationInterval(interval; offset=0)

Return a callable `IterationInterval` that "actuates" (schedules output or callback execution)
whenever the model iteration (modified by `offset`) is a multiple of `interval`.

For example,

* `IterationInterval(100)` actuates at iterations `[100, 200, 300, ...]`.
* `IterationInterval(100, offset=-1)` actuates at iterations `[99, 199, 299, ...]`.
"""
IterationInterval(interval::Int; offset=0) = IterationInterval(interval, offset)
(schedule::IterationInterval)(model) = (model.clock.iteration - schedule.offset) % schedule.interval == 0

next_actuation_time(schedule::IterationInterval) = Inf

#####
##### WallTimeInterval
#####

mutable struct WallTimeInterval{FT} <: AbstractSchedule
    interval :: FT
    previous_actuation_time :: FT
end

"""
    WallTimeInterval(interval; start_time = time_ns() * 1e-9)

Return a callable `WallTimeInterval` that schedules periodic output or diagnostic evaluation
on a `interval` of "wall time" while a simulation runs, in units of seconds.

The "wall time" is the actual real world time in seconds, as kept by an actual
or hypothetical clock hanging on your wall.

The keyword argument `start_time` can be used to specify a starting wall time
other than the moment `WallTimeInterval` is constructed.
"""
function WallTimeInterval(interval; start_time = time_ns() * 1e-9)
    FT = Oceananigans.defaults.FloatType
    return WallTimeInterval(convert(FT, interval), convert(FT, interval))
end

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

mutable struct SpecifiedTimes{FT} <: AbstractSchedule
    times :: Vector{FT}
    previous_actuation :: Int
end

"""
    SpecifiedTimes(times)

Return a callable `TimeInterval` that "actuates" (schedules output or callback execution)
whenever the model's clock equals the specified values in `times`. For example,

* `SpecifiedTimes([1, 15.3])` actuates when `model.clock.time` is `1` and `15.3`.

!!! info "Sorting specified times"
    The specified `times` need not be ordered as the `SpecifiedTimes` constructor
    will check and order them in ascending order if needed.
"""
function SpecifiedTimes(times::Vararg{T}) where T<:Number
    FT = Oceananigans.defaults.FloatType
    return SpecifiedTimes(sort([convert(FT, t) for t in times]), 0)
end

SpecifiedTimes(times) = SpecifiedTimes(times...)

function next_actuation_time(st::SpecifiedTimes)
    if st.previous_actuation >= length(st.times)
        return Inf
    else
        return st.times[st.previous_actuation+1]
    end
end

function (st::SpecifiedTimes)(model)
    current_time = model.clock.time

    if current_time >= next_actuation_time(st)
        st.previous_actuation += 1
        return true
    end

    return false
end

initialize!(st::SpecifiedTimes, model) = st(model)

function schedule_aligned_time_step(schedule::SpecifiedTimes, clock, Δt)
    δt = next_actuation_time(schedule) - clock.time
    return min(Δt, δt)
end

function specified_times_str(st)
    str_elems = ["$(prettytime(t)), " for t in st.times]
    str = string("[", str_elems...)

    # Remove final separator ", "
    str = str[1:end-2]

    # Add closing bracket
    return string(str, "]")
end

#####
##### ConsecutiveIterations
#####

mutable struct ConsecutiveIterations{S} <: AbstractSchedule
    parent :: S
    consecutive_iterations :: Int
    previous_parent_actuation_iteration :: Int
end

"""
    ConsecutiveIterations(parent_schedule)

Return a `schedule::ConsecutiveIterations` that actuates both when `parent_schedule`
actuates, and at iterations immediately following the actuation of `parent_schedule`.
This can be used, for example, when one wants to use output to reproduce a first-order approximation
of the time derivative of a quantity.
"""
ConsecutiveIterations(parent_schedule, N=1) = ConsecutiveIterations(parent_schedule, N, 0)

function (schedule::ConsecutiveIterations)(model)
    if schedule.parent(model)
        schedule.previous_parent_actuation_iteration = model.clock.iteration
        return true
    elseif model.clock.iteration - schedule.consecutive_iterations <= schedule.previous_parent_actuation_iteration
        return true # The iteration _after_ schedule.parent actuated!
    else
        return false
    end
end

schedule_aligned_time_step(schedule::ConsecutiveIterations, clock, Δt) =
    schedule_aligned_time_step(schedule.parent, clock, Δt)

#####
##### Any and AndSchedule
#####

struct AndSchedule{S} <: AbstractSchedule
    schedules :: S
    AndSchedule(schedules::S) where S <: Tuple = new{S}(schedules)
end

"""
    AndSchedule(schedules...)

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
    OrSchedule(schedules...)

Return a schedule that actuates when any of the `child_schedule`s actuates.
"""
OrSchedule(schedules...) = OrSchedule(Tuple(schedules))

function (as::OrSchedule)(model)
    # Ensure that all `schedules` get queried
    actuations = Tuple(schedule(model) for schedule in as.schedules)
    return any(actuations)
end

schedule_aligned_time_step(any_or_all_schedule::Union{OrSchedule, AndSchedule}, clock, Δt) =
    minimum(schedule_aligned_time_step(schedule, clock, Δt)
            for schedule in any_or_all_schedule.schedules)

#####
##### Show methods
#####

Base.summary(schedule::IterationInterval) = string("IterationInterval(", schedule.interval, ")")
Base.summary(schedule::TimeInterval) = string("TimeInterval(", prettytime(schedule.interval), ")")
Base.summary(schedule::SpecifiedTimes) = string("SpecifiedTimes(", specified_times_str(schedule), ")")
Base.summary(schedule::ConsecutiveIterations) = string("ConsecutiveIterations(",
                                                       summary(schedule.parent), ", ",
                                                       schedule.consecutive_iterations, ")")

