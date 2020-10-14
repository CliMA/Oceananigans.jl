using Oceananigans.Diagnostics: AbstractDiagnostic

import Oceananigans.Utils: AbstractTrigger
import Oceananigans.Fields: AbstractField, compute!
import Oceananigans.OutputWriters: fetch_output
import Oceananigans.Diagnostics: run_diagnostic!

"""
    mutable struct WindowedTimeAverageInterval <: AbstractTrigger

"""
mutable struct WindowedTimeAverageInterval <: AbstractTrigger
                  time_interval :: Float64
                    time_window :: Float64
                         stride :: Int
    previous_interval_stop_time :: Float64
                     collecting :: Bool
end

WindowedTimeAverageInterval(; interval, window, stride=1) =
    WindowedTimeAverageInterval(Float64(interval), Float64(window), stride, 0.0, false)

(trigger::WindowedTimeAverageInterval)(model) =
    trigger.collecting || model.clock.time >= trigger.previous_interval_stop_time + trigger.time_interval - trigger.time_window

"""
    WindowedTimeAverage{R, OP, FS} <: AbstractDiagnostic

An object for computing 'windowed' time averages, or moving time-averages
of a `operand` over a specified `time_window`, collected on `time_interval`.
"""
mutable struct WindowedTimeAverage{R, OP, FS} <: AbstractDiagnostic
                         result :: R
                        operand :: OP
              window_start_time :: Float64
         window_start_iteration :: Int
       previous_collection_time :: Float64
                   field_slicer :: FS
                        trigger :: WindowedTimeAverageInterval
end

"""
    WindowedTimeAverage(operand, model=nothing; trigger, field_slicer = FieldSlicer())
                                                        
Returns an object for computing running averages of `operand` over `trigger.time_window`,
recurring on `trigger.time_interval`. During the collection period, averages are computed
every `trigger.stride` iteration. 

`operand` may be an `Oceananigans.Field`, `Oceananigans.AbstractOperations.Computation,
or `Oceananigans.Diagnostics.Average`.

Calling `wta(model)` for `wta::WindowedTimeAverage` object returns `wta.result`.
""" 
function WindowedTimeAverage(operand, model=nothing; trigger, field_slicer = FieldSlicer())
                                                     
    output = fetch_output(operand, model, field_slicer)
    result = similar(output) # convert views to arrays
    result .= output # initialize `result` with initial output

    return WindowedTimeAverage(result, operand, 0.0, 0, 0.0, field_slicer, stride, trigger)
end

function accumulate_result!(wta, model)

    # Time increment:
    Δt = model.clock.time - wta.previous_collection_time    

    # Time intervals:
    T_current = model.clock.time - wta.window_start_time
    T_previous = wta.previous_collection_time - wta.window_start_time

    # Accumulate left Riemann sum
    integrand = fetch_output(wta.operand, model, wta.field_slicer)
    @. wta.result = (wta.result * T_previous + integrand * Δt) / T_current

    # Save time of integrand collection
    wta.previous_collection_time = model.clock.time
                    
    return nothing
end

function run_diagnostic!(wta::WindowedTimeAverage, model)

    if model.clock.iteration == 0 # initialize previous interval stop time
        wta.trigger.previous_interval_stop_time = model.clock.time
    end

    # Don't start collecting if we are *only* "initializing" run_diagnostic! at the beginning
    # of a Simulation.
    #
    # implies we are always collecting)
    
    initializing = model.clock.iteration == 0 &&
        model.clock.time < wta.trigger.previous_interval_stop_time + wta.trigger.time_interval - wta.trigger.time_window

    if !(wta.trigger.collecting) && !(initializing)
        # run_diagnostic! has been called, but we are not currently collecting data.
        # Initialize data collection:

        # Start averaging period
        wta.trigger.collecting = true

        # Zero out result
        wta.result .= 0

        # Save averaging start time and the initial data collection time
        wta.window_start_time = model.clock.time
        wta.window_start_iteration = model.clock.iteration
        wta.trigger.previous_collection_time = model.clock.time

    elseif model.clock.time >= wta.previous_interval_stop_time + wta.time_interval
        # Output is imminent. Finalize averages and cease data collection.
        accumulate_result!(wta, model)

        # Averaging period is complete.
        wta.trigger.collecting = false

        # Reset the "previous" interval time,
        # subtracting a sliver that presents window overshoot from accumulating.
        wta.trigger.previous_interval_stop_time = model.clock.time - rem(model.clock.time, wta.trigger.time_interval)

    elseif mod(model.clock.iteration - wta.window_start_iteration, wta.trigger.stride) == 0
        # Collect data as usual
        accumulate_result!(wta, model)
    end

    return nothing
end

function (wta::WindowedTimeAverage)(model)

    # For the paranoid
    wta.collecting && 
        model.clock.iteration > 0 && 
        @warn "Returning a WindowedTimeAverage before the collection period is complete."

    return wta.result
end
