using Oceananigans.Utils: AbstractSchedule, prettytime
using Oceananigans.TimeSteppers: Clock
using Dates: Period, Second, value

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

determine_epsilon(eltype) = 0
determine_epsilon(::Type{T}) where T <: AbstractFloat = eps(T)
determine_epsilon(::Type{<:Period}) = Second(0)

const NumberTypeWindows = Union{Number, Vector{<:Number}}
const PeriodTypeWindows = Union{Period, Vector{<:Period}}
validate_windows(times, window) = nothing  # Fallback method

function validate_windows(times, window::NumberTypeWindows)
    tol = 100 * determine_epsilon(eltype(times))

    gaps = diff(vcat(0, times))  # Prepend 0 to check first window against t=0
    any(gaps .- window .< -tol) && throw(ArgumentError("Averaging windows overlap: some gaps between specified times are less than the window size."))

    return nothing
end

function validate_windows(times, window::PeriodTypeWindows)
    if length(times) >= 2
        window_starts = times .- window
        prev_window_ends = times[1:end-1]
        any(window_starts[2:end] .< prev_window_ends) && throw(ArgumentError("Averaging windows overlap: some gaps between specified times are less than the window size."))
    end

    # Note: We cannot check if the first window extends before the simulation start
    # because the model clock is not available at construction time

    return nothing
end

function AveragedSpecifiedTimes(times, window::Vector; kw...)
    length(window) == length(times) || throw(ArgumentError("When providing a vector of windows, its length $(length(window)) must match the number of specified times $(length(times))."))
    perm = sortperm(times)
    sorted_times = times[perm]
    sorted_window = window[perm]

    # Check for overlapping windows
    validate_windows(sorted_times, sorted_window)

    return AveragedSpecifiedTimes(SpecifiedTimes(sorted_times); window=sorted_window, kw...)
end

function AveragedSpecifiedTimes(times, window; kw...)
    specified_times = SpecifiedTimes(times)

    # Check for overlapping windows (scalar window case)
    if length(specified_times.times) > 1
        validate_windows(specified_times.times, window)
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

initialize_schedule!(sch::AveragedSpecifiedTimes, clock) = nothing

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
