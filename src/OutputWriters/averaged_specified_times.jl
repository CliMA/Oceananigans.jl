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

"""
    AveragedSpecifiedTimes(times; window, stride=1)
    AveragedSpecifiedTimes(specified_times::SpecifiedTimes; window, stride=1)

Returns a `schedule` that specifies time-averaging of output at specified times.
The time `window` specifies the extent of the time-average that precedes each
specified output time.

`output` is computed and accumulated into the average every `stride` iterations
during the averaging window. For example, `stride=1` computes output every iteration,
whereas `stride=2` computes output every other iteration. Time-averages with
longer `stride`s are faster to compute, but less accurate.

The `times` can be specified as:
- Numbers (interpreted as seconds)
- `DateTime` objects
- `Period` objects (e.g., `Day`, `Hour`)

The `window` can be:
- A single value (scalar) applying to all specified times
- A vector of values with length matching `times`, allowing different averaging windows for each output time

Examples
========

Basic usage with constant window
---------------------------------

```jldoctest averaged_specified_times
using Oceananigans.OutputWriters: AveragedSpecifiedTimes

schedule = AveragedSpecifiedTimes([4.0, 8.0, 12.0], window=2.0)

# output
AveragedSpecifiedTimes(window=2 seconds, stride=1, specified_times=SpecifiedTimes([4 seconds, 8 seconds, 12 seconds]))
```

An `AveragedSpecifiedTimes` schedule directs an output writer
to time-average its outputs before writing them to disk:

```@example averaged_specified_times
using Oceananigans
using Oceananigans.Units

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=10minutes, stop_time=30days)

simulation.output_writers[:velocities] = JLD2Writer(model, model.velocities,
                                                    filename="averaged_velocity_data.jld2",
                                                    schedule=AveragedSpecifiedTimes([4days, 8days, 12days], window=2days, stride=2))
```

Varying windows
---------------

Different averaging windows can be specified for each output time:

```jldoctest averaged_specified_times_varying
using Oceananigans.OutputWriters: AveragedSpecifiedTimes
using Oceananigans.Units

# Different windows for each output time
schedule = AveragedSpecifiedTimes([4days, 8days, 12days], window=[1day, 2days, 3days])

# output
AveragedSpecifiedTimes(window=[1 day, 2 days, 3 days], stride=1, specified_times=SpecifiedTimes([4 days, 8 days, 12 days]))
```

Using DateTime
--------------

For simulations with calendar-based timing:

```jldoctest averaged_specified_times_datetime
using Oceananigans.OutputWriters: AveragedSpecifiedTimes
using Dates

# Specify times as DateTime objects
times = [DateTime(2024, 1, 5), DateTime(2024, 1, 10), DateTime(2024, 1, 15)]
schedule = AveragedSpecifiedTimes(times, window=Day(2))

# output
AveragedSpecifiedTimes(window=2 days, stride=1, specified_times=SpecifiedTimes([2024-01-05T00:00:00, 2024-01-10T00:00:00, 2024-01-15T00:00:00]))
```

Using Periods
-------------

Period types can be used for more readable time specifications:

```jldoctest averaged_specified_times_periods
using Oceananigans.OutputWriters: AveragedSpecifiedTimes
using Dates

# Specify times and windows using Period types
schedule = AveragedSpecifiedTimes([Day(4), Day(8), Day(12)], window=Day(2))

# output
AveragedSpecifiedTimes(window=2 days, stride=1, specified_times=SpecifiedTimes([4 days, 8 days, 12 days]))
```

Varying Period windows
----------------------

Different Period windows can also be specified:

```jldoctest averaged_specified_times_varying_periods
using Oceananigans.OutputWriters: AveragedSpecifiedTimes
using Dates

# Different Period windows for each output time
schedule = AveragedSpecifiedTimes([Day(4), Day(8), Day(12)], window=[Hour(12), Day(1), Day(2)])

# output
AveragedSpecifiedTimes(window=[12 hours, 24 hours, 48 hours], stride=1, specified_times=SpecifiedTimes([4 days, 8 days, 12 days]))
```
"""
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

function prognostic_state(schedule::AveragedSpecifiedTimes)
    return (specified_times = prognostic_state(schedule.specified_times),
            collecting = schedule.collecting)
end

function restore_prognostic_state!(schedule::AveragedSpecifiedTimes, state)
    restore_prognostic_state!(schedule.specified_times, state.specified_times)
    schedule.collecting = state.collecting
    return schedule
end

restore_prognostic_state!(::AveragedSpecifiedTimes, ::Nothing) = nothing

TimeInterval(sch::AveragedSpecifiedTimes) = TimeInterval(sch.specified_times.times)
Base.copy(sch::AveragedSpecifiedTimes) = AveragedSpecifiedTimes(copy(sch.specified_times); window=sch.window, stride=sch.stride)

next_actuation_time(sch::AveragedSpecifiedTimes) = Oceananigans.Utils.next_actuation_time(sch.specified_times)

# Helper function to format window for display
function format_window(window)
    if window isa Vector
        # Format each element and join with ", " inside brackets
        elements = [prettytime(w) for w in window]
        return "[" * join(elements, ", ") * "]"
    else
        return prettytime(window)
    end
end

Base.show(io::IO, schedule::AveragedSpecifiedTimes) = print(io, summary(schedule))

Base.summary(schedule::AveragedSpecifiedTimes) = string("AveragedSpecifiedTimes(",
                                                        "window=", format_window(schedule.window), ", ",
                                                        "stride=", schedule.stride, ", ",
                                                        "specified_times=", schedule.specified_times,  ")")

show_averaging_schedule(schedule::AveragedSpecifiedTimes) = string(" averaged on ", summary(schedule))
