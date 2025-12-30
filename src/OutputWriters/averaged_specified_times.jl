using Oceananigans.Utils: AbstractSchedule, prettytime
using Oceananigans.TimeSteppers: Clock
using Dates: Period, Second, value, DateTime

import Oceananigans: initialize!
import Oceananigans.Utils: TimeInterval, SpecifiedTimes

"""
    mutable struct AveragedSpecifiedTimes <: AbstractSchedule

A schedule for averaging over windows that precede SpecifiedTimes.
"""
mutable struct AveragedSpecifiedTimes{S, W} <: AbstractSchedule
    specified_times :: S
    window :: W
    stride :: Int
    collecting :: Bool
end

const VaryingWindowAveragedSpecifiedTimes = AveragedSpecifiedTimes{<:Any, <:Vector}

AveragedSpecifiedTimes(specified_times::SpecifiedTimes; window, stride=1) =
    AveragedSpecifiedTimes(specified_times, window, stride, false)

AveragedSpecifiedTimes(times; window, kw...) = AveragedSpecifiedTimes(times, window; kw...)

determine_epsilon(eltype) = 0 # Fallback method
determine_epsilon(::Type{T}) where T <: AbstractFloat = eps(T)
determine_epsilon(::Type{<:Period}) = Second(0)
determine_epsilon(::Type{<:DateTime}) = Second(0)

const NumberTypeWindows = Union{Number, Vector{<:Number}}
const PeriodTypeWindows = Union{Period, Vector{<:Period}}
validate_nonoverlapping_windows(times, window) = nothing  # Fallback method

function validate_nonoverlapping_windows(times, window::NumberTypeWindows)
    # Check for overlapping windows between consecutive times
    # Works for both scalar and vector windows through broadcasting
    if length(times) >= 2
        tol = 100 * determine_epsilon(eltype(times))
        window_starts = times .- window  # Broadcasts correctly for both scalar and vector
        prev_times = times[1:end-1]
        if any(window_starts[2:end] .- prev_times .< -tol)
            throw(ArgumentError("Averaging windows overlap: some windows start before the previous specified time."))
        end
    end

    # Note: Validation of the first window against the simulation start time
    # happens at runtime in validate_schedule_runtime().

    return nothing
end

function validate_nonoverlapping_windows(times, window::PeriodTypeWindows)
    # Check for overlapping windows between consecutive times
    # Works for both scalar and vector Period windows through broadcasting
    if length(times) >= 2
        window_starts = times .- window  # Broadcasts correctly for both scalar and vector
        prev_times = times[1:end-1]
        any(window_starts[2:end] .< prev_times) && throw(ArgumentError("Averaging windows overlap: some windows start before the previous specified time."))
    end

    # Note: Validation of the first window against the simulation start time
    # happens at runtime in validate_schedule_runtime().

    return nothing
end

function AveragedSpecifiedTimes(times, window::Vector; kw...)
    length(window) == length(times) || throw(ArgumentError("When providing a vector of windows, its length $(length(window)) must match the number of specified times $(length(times))."))
    perm = sortperm(times)
    sorted_times = times[perm]
    sorted_window = window[perm]

    # Check for overlapping windows
    validate_nonoverlapping_windows(sorted_times, sorted_window)

    return AveragedSpecifiedTimes(SpecifiedTimes(sorted_times); window=sorted_window, kw...)
end

function AveragedSpecifiedTimes(times, window; kw...)
    specified_times = SpecifiedTimes(times)

    # Check for overlapping windows (scalar window case)
    if length(specified_times.times) > 1
        validate_nonoverlapping_windows(specified_times.times, window)
    end

    return AveragedSpecifiedTimes(specified_times; window, kw...)
end

get_next_window(schedule::VaryingWindowAveragedSpecifiedTimes) = schedule.window[schedule.specified_times.previous_actuation + 1]
get_next_window(schedule) = schedule.window

function (schedule::AveragedSpecifiedTimes)(model)
    time = model.clock.time

    next = schedule.specified_times.previous_actuation + 1
    next > length(schedule.specified_times.times) && return false

    next_time = schedule.specified_times.times[next]
    window = get_next_window(schedule)

    schedule.collecting || time >= next_time - window
end

"""
    validate_schedule_runtime(schedule::AveragedSpecifiedTimes, clock)

Validate that the first averaging window does not extend before the simulation start time.
This validation can only be performed at runtime when the model clock is available.
"""
function validate_schedule_runtime(schedule::AveragedSpecifiedTimes, clock)
    # Only validate if we haven't started collecting or completed any actuations yet
    (schedule.specified_times.previous_actuation > 0 || schedule.collecting) && return nothing

    # Get the first specified time and window
    isempty(schedule.specified_times.times) && return nothing
    first_time = first(schedule.specified_times.times)
    first_window = schedule.window isa Vector ? first(schedule.window) : schedule.window

    # Check if the first window extends before the simulation start time
    window_start = first_time - first_window

    tol = 100 * determine_epsilon(typeof(clock.time))

    if window_start < clock.time - tol
        throw(ArgumentError("The first averaging window starts at $(prettytime(window_start)), " *
                            "which is before the simulation start time $(prettytime(clock.time)). " *
                            "Consider adjusting the first specified time or reducing the window size."))
    end

    return nothing
end

initialize!(sch::AveragedSpecifiedTimes, model) = validate_schedule_runtime(sch, model.clock)

function outside_window(schedule::AveragedSpecifiedTimes, clock)
    next = schedule.specified_times.previous_actuation + 1
    next > length(schedule.specified_times.times) && return true
    next_time = schedule.specified_times.times[next]
    window = get_next_window(schedule)
    return clock.time <= next_time - window
end

function end_of_window(schedule::AveragedSpecifiedTimes, clock)
    next = schedule.specified_times.previous_actuation + 1
    next > length(schedule.specified_times.times) && return true
    next_time = schedule.specified_times.times[next]
    return clock.time >= next_time
end

TimeInterval(sch::AveragedSpecifiedTimes) = TimeInterval(sch.specified_times.times)
Base.copy(sch::AveragedSpecifiedTimes) = AveragedSpecifiedTimes(copy(sch.specified_times); window=sch.window, stride=sch.stride)

next_actuation_time(sch::AveragedSpecifiedTimes) = Oceananigans.Utils.next_actuation_time(sch.specified_times)

Base.summary(schedule::AveragedSpecifiedTimes) = string("AveragedSpecifiedTimes(",
                                                        "window=", prettytime(schedule.window), ", ",
                                                        "stride=", schedule.stride, ", ",
                                                        "times=", schedule.specified_times,  ")")

show_averaging_schedule(schedule::AveragedSpecifiedTimes) = string(" averaged on ", summary(schedule))
