using Oceananigans.Diagnostics: AbstractDiagnostic
using Oceananigans.OutputWriters: fetch_output
using Oceananigans.Utils: AbstractSchedule, prettytime

import Oceananigans: run_diagnostic!
import Oceananigans.Utils: TimeInterval
import Oceananigans.Fields: location, indices

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

`` ⟨a⟩ = 1/T \\int_{tᵢ-T}^T a \\mathrm{d} t ,``

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
├── array type: Array{Float32}
├── including: [:grid, :coriolis, :buoyancy, :closure]
└── max filesize: Inf YiB
```
"""
function AveragedTimeInterval(interval; window=interval, stride=1)
    window > interval && throw(ArgumentError("Averaging window $window is greater than the output interval $interval."))
    return AveragedTimeInterval(Float64(interval), Float64(window), stride, 0.0, false)
end

# Determines whether or not to call run_diagnostic
(schedule::AveragedTimeInterval)(model) =
    schedule.collecting || model.clock.time >= schedule.previous_interval_stop_time + schedule.interval - schedule.window

TimeInterval(schedule::AveragedTimeInterval) = TimeInterval(schedule.interval)

Base.copy(schedule::AveragedTimeInterval) = AveragedTimeInterval(schedule.interval, window=schedule.window, stride=schedule.stride)

"""
    WindowedTimeAverage{OP, R, FS} <: AbstractDiagnostic

An object for computing 'windowed' time averages, or moving time-averages
of a `operand` over a specified `window`, collected on `interval`.
"""
mutable struct WindowedTimeAverage{OP, R} <: AbstractDiagnostic
                      result :: R
                     operand :: OP
           window_start_time :: Float64
      window_start_iteration :: Int
    previous_collection_time :: Float64
                    schedule :: AveragedTimeInterval
end

"""
    WindowedTimeAverage(operand, model=nothing; schedule)

Returns an object for computing running averages of `operand` over `schedule.window` and
recurring on `schedule.interval`, where `schedule` is an `AveragedTimeInterval`.
During the collection period, averages are computed every `schedule.stride` iteration.

`operand` may be a `Oceananigans.Field` or a function that returns an array or scalar.

Calling `wta(model)` for `wta::WindowedTimeAverage` object returns `wta.result`.
"""
function WindowedTimeAverage(operand, model=nothing; schedule)

    output = fetch_output(operand, model)
    result = similar(output) # convert views to arrays, for example
    result .= output # initialize `result` with initial output

    return WindowedTimeAverage(result, operand, 0.0, 0, 0.0, schedule)
end

# Time-averaging doesn't change spatial location
location(wta::WindowedTimeAverage) = location(wta.operand)
indices(wta::WindowedTimeAverage) = indices(wta.operand)

function accumulate_result!(wta, model)

    # Time increment:
    Δt = model.clock.time - wta.previous_collection_time

    # Time intervals:
    T_current = model.clock.time - wta.window_start_time
    T_previous = wta.previous_collection_time - wta.window_start_time

    # Accumulate left Riemann sum
    integrand = fetch_output(wta.operand, model)

    @. wta.result = (wta.result * T_previous + integrand * Δt) / T_current

    # Save time of integrand collection
    wta.previous_collection_time = model.clock.time

    return nothing
end

function run_diagnostic!(wta::WindowedTimeAverage, model)

    if model.clock.iteration == 0 # initialize previous interval stop time
        wta.schedule.previous_interval_stop_time = model.clock.time
    end

    # Don't start collecting if we are *only* "initializing" run_diagnostic! at the beginning
    # of a Simulation.
    #
    # Note: this can be false at the zeroth iteration if interval == window (which
    # implies we are always collecting)

    unscheduled = model.clock.iteration == 0 &&
        model.clock.time < wta.schedule.previous_interval_stop_time + wta.schedule.interval - wta.schedule.window

    if unscheduled
        # This is an "unscheduled" call to run_diagnostic! --- which occurs when run_diagnostic!
        # is called at the beginning of a run (and schedule.interval != schedule.window).
        # In this case we do nothing.
    elseif !(wta.schedule.collecting)
        # run_diagnostic! has been called on schedule, but we are not currently collecting data.
        # Initialize data collection:

        # Start averaging period
        wta.schedule.collecting = true

        # Zero out result
        wta.result .= 0

        # Save averaging start time and the initial data collection time
        wta.window_start_time = model.clock.time
        wta.window_start_iteration = model.clock.iteration
        wta.previous_collection_time = model.clock.time

    elseif model.clock.time >= wta.schedule.previous_interval_stop_time + wta.schedule.interval
        # Output is imminent. Finalize averages and cease data collection.
        accumulate_result!(wta, model)

        # Averaging period is complete.
        wta.schedule.collecting = false

        # Reset the "previous" interval time, subtracting a sliver that presents overshoot from accumulating.
        wta.schedule.previous_interval_stop_time = model.clock.time - rem(model.clock.time, wta.schedule.interval)

    elseif mod(model.clock.iteration - wta.window_start_iteration, wta.schedule.stride) == 0
        # Collect data as usual
        accumulate_result!(wta, model)
    end

    return nothing
end

function (wta::WindowedTimeAverage)(model)

    # For the paranoid
    wta.schedule.collecting &&
        model.clock.iteration > 0 &&
        @warn "Returning a WindowedTimeAverage before the collection period is complete."

    return wta.result
end

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

