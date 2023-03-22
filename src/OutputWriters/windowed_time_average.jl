using Oceananigans.Diagnostics: AbstractDiagnostic
using Oceananigans.OutputWriters: fetch_output
using Oceananigans.Models: AbstractModel
using Oceananigans.Utils: AbstractSchedule, prettytime
using Oceananigans.TimeSteppers: Clock

import Oceananigans: run_diagnostic!
import Oceananigans.Utils: TimeInterval, SpecifiedTimes
import Oceananigans.Fields: location, indices, set_interior!

"""
    mutable struct AveragedTimeInterval <: AbstractSchedule

Container for parameters that configure and handle time-averaged output.
"""
mutable struct AveragedTimeInterval <: AbstractSchedule
    interval :: Float64
    window :: Float64
    stride :: Int
    previous_interval_stop_time :: Float64
    collecting :: Bool
end

"""
    AveragedTimeInterval(interval; window=interval, stride=1)

Returns a `schedule` that specifies periodic time-averaging of output.
The time `window` specifies the extent of the time-average, which
reoccurs every `interval`.

`output` is computed and accumulated into the average every `stride` iterations
during the averaging window. For example, `stride=1` computs output every iteration,
whereas `stride=2` computes output every other iteration. Time-averages with
longer `stride`s are faster to compute, but less accurate.

The time-average of ``a`` is a left Riemann sum corresponding to

```math
⟨a⟩ = T⁻¹ \\int_{tᵢ-T}^{tᵢ} a \\mathrm{d} t \\, ,
```

where ``⟨a⟩`` is the time-average of ``a``, ``T`` is the time-window for averaging,
and the ``tᵢ`` are discrete times separated by the time `interval`. The ``tᵢ`` specify
both the end of the averaging window and the time at which output is written.

Example
=======

```jldoctest averaged_time_interval
using Oceananigans.OutputWriters: AveragedTimeInterval
using Oceananigans.Utils: year, years

schedule = AveragedTimeInterval(4years, window=1year)

# output
AveragedTimeInterval(window=1 year, stride=1, interval=4 years)
```

An `AveragedTimeInterval` schedule directs an output writer
to time-average its outputs before writing them to disk:

```jldoctest averaged_time_interval
using Oceananigans
using Oceananigans.OutputWriters: JLD2OutputWriter
using Oceananigans.Utils: minutes

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=10minutes, stop_time=30years)

simulation.output_writers[:velocities] = JLD2OutputWriter(model, model.velocities,
                                                          filename= "averaged_velocity_data.jld2",
                                                          schedule = AveragedTimeInterval(4years, window=1year, stride=2))

# output
JLD2OutputWriter scheduled on TimeInterval(4 years):
├── filepath: ./averaged_velocity_data.jld2
├── 3 outputs: (u, v, w) averaged on AveragedTimeInterval(window=1 year, stride=2, interval=4 years)
├── array type: Array{Float64}
├── including: [:grid, :coriolis, :buoyancy, :closure]
└── max filesize: Inf YiB
```
"""
function AveragedTimeInterval(interval; window=interval, stride=1)
    window > interval && throw(ArgumentError("Averaging window $window is greater than the output interval $interval."))
    return AveragedTimeInterval(Float64(interval), Float64(window), stride, 0.0, false)
end

# Schedule actuation
(sch::AveragedTimeInterval)(model) = sch.collecting || model.clock.time >= sch.previous_interval_stop_time + sch.interval - sch.window
initialize_schedule!(sch::AveragedTimeInterval, clock) = sch.previous_interval_stop_time = clock.time - rem(clock.time, sch.interval)
outside_window(sch::AveragedTimeInterval, clock) = clock.time <  sch.previous_interval_stop_time + sch.interval - sch.window   
end_of_window(sch::AveragedTimeInterval, clock) = clock.time >= sch.previous_interval_stop_time + sch.interval

TimeInterval(schedule::AveragedTimeInterval) = TimeInterval(schedule.interval)
Base.copy(sch::AveragedTimeInterval) = AveragedTimeInterval(sch.interval, window=sch.window, stride=sch.stride)

"""
    mutable struct AveragedSpecifiedTimes <: AbstractSchedule

A schedule for averaging over windows that precede SpecifiedTimes.
"""
mutable struct AveragedSpecifiedTimes <: AbstractSchedule
    specified_times :: SpecifiedTimes
    window :: Float64
    stride :: Int
    collecting :: Bool
end

AveragedSpecifiedTimes(specified_times::SpecifiedTimes; window, stride=1) =
    AveragedSpecifiedTimes(specified_times, window, stride, false)

AveragedSpecifiedTimes(times; kw...) = AveragedSpecifiedTimes(SpecifiedTimes(times); kw...)

# Schedule actuation
function (schedule::AveragedSpecifiedTimes)(model)
    time = model.clock.time

    next = schedule.specified_times.previous_actuation + 1
    next > length(schedule.specified_times.times) && return false

    next_time = schedule.specified_times.times[next]
    window = schedule.window

    schedule.collecting || time >= next_time - window
end

initialize_schedule!(sch::AveragedSpecifiedTimes, clock) = nothing

function outside_window(schedule::AveragedSpecifiedTimes, clock)
    next = schedule.specified_times.previous_actuation + 1
    next > length(schedule.specified_times.times) && return true
    next_time = schedule.specified_times.times[next]
    return clock.time < next_time - schedule.window
end

function end_of_window(schedule::AveragedSpecifiedTimes, clock)
    next = schedule.specified_times.previous_actuation + 1
    next > length(schedule.specified_times.times) && return true
    next_time = schedule.specified_times.times[next]
    return clock.time >= next_time
end

#####
##### WindowedTimeAverage
#####

mutable struct WindowedTimeAverage{OP, R, S} <: AbstractDiagnostic
                      result :: R
                     operand :: OP
           window_start_time :: Float64
      window_start_iteration :: Int
    previous_collection_time :: Float64
                    schedule :: S
               fetch_operand :: Bool
end

const IntervalWindowedTimeAverage = WindowedTimeAverage{<:Any, <:Any, <:AveragedTimeInterval}
const SpecifiedWindowedTimeAverage = WindowedTimeAverage{<:Any, <:Any, <:AveragedSpecifiedTimes}

stride(wta::IntervalWindowedTimeAverage) = wta.schedule.stride
stride(wta::SpecifiedWindowedTimeAverage) = wta.schedule.stride

"""
    WindowedTimeAverage(operand, model=nothing; schedule)

Returns an object for computing running averages of `operand` over `schedule.window` and
recurring on `schedule.interval`, where `schedule` is an `AveragedTimeInterval`.
During the collection period, averages are computed every `schedule.stride` iteration.

`operand` may be a `Oceananigans.Field` or a function that returns an array or scalar.

Calling `wta(model)` for `wta::WindowedTimeAverage` object returns `wta.result`.
"""
function WindowedTimeAverage(operand, model=nothing; schedule, fetch_operand=true)

    if fetch_operand
        output = fetch_output(operand, model)
        result = similar(output)
        result .= output
    else
        result = similar(operand)
        result .= operand
    end
        
    return WindowedTimeAverage(result, operand, 0.0, 0, 0.0, schedule, fetch_operand)
end

# Time-averaging doesn't change spatial location
location(wta::WindowedTimeAverage) = location(wta.operand)
indices(wta::WindowedTimeAverage) = indices(wta.operand)
set_interior!(u::Field, wta::WindowedTimeAverage) = set_interior!(u, wta.result)
Base.parent(wta::WindowedTimeAverage) = parent(wta.result)

# This is called when output is requested.
function (wta::WindowedTimeAverage)(model)

    # For the paranoid
    wta.schedule.collecting &&
        model.clock.iteration > 0 &&
        @warn "Returning a WindowedTimeAverage before the collection period is complete."

    return wta.result
end

function accumulate_result!(wta, model::AbstractModel)
    integrand = wta.fetch_operand ? fetch_output(wta.operand, model) : wta.operand
    return accumulate_result!(wta, model.clock, integrand)
end

function accumulate_result!(wta, clock::Clock, integrand=wta.operand)
    # Time increment:
    Δt = clock.time - wta.previous_collection_time

    # Time intervals:
    T_current = clock.time - wta.window_start_time
    T_previous = wta.previous_collection_time - wta.window_start_time

    # Accumulate left Riemann sum
    @. wta.result = (wta.result * T_previous + integrand * Δt) / T_current

    # Save time of integrand collection
    wta.previous_collection_time = clock.time

    return nothing
end

function advance_time_average!(wta::WindowedTimeAverage, model)

    if model.clock.iteration == 0 # initialize previous interval stop time
        initialize_schedule!(wta.schedule, model.clock)
    end

    # Don't start collecting if we are *only* "initializing" at the beginning
    # of a Simulation.
    #
    # Note: this can be false at the zeroth iteration if interval == window (which
    # implies we are always collecting)

    unscheduled = model.clock.iteration == 0 && outside_window(wta.schedule, model.clock)

    if unscheduled
        # This is an "unscheduled" call to run_diagnostic! --- which occurs when run_diagnostic!
        # is called at the beginning of a run (and schedule.interval != schedule.window).
        # In this case we do nothing.
        
    # Next, we handle WindowedTimeAverage's two "modes": collecting, and not collecting.    
    #
    # The "not collecting" mode indicates that "initialization" is needed before collecting.
    # eg the result has to be zeroed prior to starting to collect. However, because we accumulate
    # a time-average with a left Riemann sum, we do not do any calculations during initialization.
    #
    # The "collecting" mode indicates that collecting is occuring; therefore we accumulate the time-average.
    
    elseif !(wta.schedule.collecting)
        # run_diagnostic! has been called on schedule but we are not currently collecting data.
        # Initialize data collection:

        # Start averaging period
        wta.schedule.collecting = true

        # Zero out result
        wta.result .= 0

        # Save averaging start time and the initial data collection time
        wta.window_start_time = model.clock.time
        wta.window_start_iteration = model.clock.iteration
        wta.previous_collection_time = model.clock.time

    elseif end_of_window(wta.schedule, model.clock)
        # Output is imminent. Finalize averages and cease data collection.
        # Note that this may induce data collecting more frequently than proscribed
        # by `wta.schedule.stride`.
        accumulate_result!(wta, model)

        # Averaging period is complete.
        wta.schedule.collecting = false

        # Reset the "previous" interval time, subtracting a sliver that presents overshoot from accumulating.
        initialize_schedule!(wta.schedule, model.clock)

    elseif mod(model.clock.iteration - wta.window_start_iteration, stride(wta)) == 0
        # Collect data as usual
        accumulate_result!(wta, model)
    end

    return nothing
end

# So it can be used as a Diagnostic
run_diagnostic!(wta::WindowedTimeAverage, model) = advance_time_average!(wta, model)

Base.show(io::IO, schedule::AveragedTimeInterval) = print(io, summary(schedule))

Base.summary(schedule::AveragedTimeInterval) = string("AveragedTimeInterval(",
                                                      "window=", prettytime(schedule.window), ", ",
                                                      "stride=", schedule.stride, ", ",
                                                      "interval=", prettytime(schedule.interval),  ")")

show_averaging_schedule(schedule) = ""
show_averaging_schedule(schedule::AveragedTimeInterval) = string(" averaged on ", summary(schedule))

output_averaging_schedule(output::WindowedTimeAverage) = output.schedule

#####
##### Utils for OutputWriters
#####
 
time_average_outputs(schedule, outputs, model) = schedule, outputs # fallback

"""
    time_average_outputs(schedule::AveragedTimeInterval, outputs, model, field_slicer)

Wrap each `output` in a `WindowedTimeAverage` on the time-averaged `schedule` and with `field_slicer`.

Returns the `TimeInterval` associated with `schedule` and a `NamedTuple` or `Dict` of the wrapped
outputs.
"""
function time_average_outputs(schedule::AveragedTimeInterval, outputs::Dict, model)
    averaged_outputs = Dict(name => WindowedTimeAverage(output, model; schedule=copy(schedule))
                            for (name, output) in outputs)

    return TimeInterval(schedule), averaged_outputs
end

function time_average_outputs(schedule::AveragedTimeInterval, outputs::NamedTuple, model)
    averaged_outputs = NamedTuple(name => WindowedTimeAverage(outputs[name], model; schedule=copy(schedule))
                                  for name in keys(outputs))

    return TimeInterval(schedule), averaged_outputs
end

