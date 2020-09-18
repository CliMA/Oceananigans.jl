using Oceananigans.Diagnostics: AbstractDiagnostic

import Oceananigans.Utils: time_to_run
import Oceananigans.Fields: AbstractField, compute!
import Oceananigans.OutputWriters: fetch_output
import Oceananigans.Diagnostics: run_diagnostic

"""
    WindowedTimeAverage{R, OP, FS} <: AbstractDiagnostic

An object for computing 'windowed' time averages, or moving time-averages
of a `operand` over a specified `time_window`, collected on `time_interval`.
"""
mutable struct WindowedTimeAverage{R, OP, FS} <: AbstractDiagnostic
                         result :: R
                        operand :: OP
                    time_window :: Float64
                  time_interval :: Float64
                         stride :: Int
                   field_slicer :: FS
              window_start_time :: Float64
         window_start_iteration :: Int
       previous_collection_time :: Float64
    previous_interval_stop_time :: Float64
                     collecting :: Bool
end

"""
    WindowedTimeAverage(operand; time_window,
                                 time_interval,
                                        stride = 1,
                                  field_slicer = FieldSlicer())
                                                        
Returns an object for computing running averages of `operand` over `time_window`,
recurring on `time_interval`. During the collection period, averages are computed
every `stride` iteration. 

`operand` may be an `Oceananigans.Field`, `Oceananigans.AbstractOperations.Computation,
or `Oceananigans.Diagnostics.Average`.

Calling `wta(model)` for `wta::WindowedTimeAverage` object returns `wta.result`.
""" 
function WindowedTimeAverage(operand, model=nothing; time_window, time_interval, stride=1,
                                                     field_slicer = FieldSlicer())

    output = fetch_output(operand, model, field_slicer)
    result = similar(output) # convert views to arrays
    result .= output # initialize `result` with initial output

    return WindowedTimeAverage(result,
                               operand,
                               time_window,
                               time_interval,
                               stride,
                               field_slicer,
                               0.0, 0, 0.0, 0.0, false)
end

function time_to_run(clock, wta::WindowedTimeAverage)
    if (wta.collecting || 
        clock.time >= wta.previous_interval_stop_time + wta.time_interval - wta.time_window)

        return true
    else
        return false
    end
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

function run_diagnostic(model, wta::WindowedTimeAverage)

    if model.clock.iteration == 0 # initialize previous interval stop time
        wta.previous_interval_stop_time = model.clock.time
    end

    # Don't start collecting if we are *only* "initializing" run_diagnostic at the beginning
    # of a Simulation.
    #
    # Note: this can be false at the first iteration if time_interval == time_window (which
    # implies we are always collecting)
    
    initializing = model.clock.iteration == 0 &&
        model.clock.time < wta.previous_interval_stop_time + wta.time_interval - wta.time_window

    if !(wta.collecting) && !(initializing)
        # run_diagnostic has been called, but we are not currently collecting data.
        # Initialize data collection:

        # Start averaging period
        wta.collecting = true

        # Zero out result
        wta.result .= 0

        # Save averaging start time and the initial data collection time
        wta.window_start_time = model.clock.time
        wta.window_start_iteration = model.clock.iteration
        wta.previous_collection_time = model.clock.time

    elseif model.clock.time - wta.window_start_time >= wta.time_window 
        # The averaging window has been exceeded. Finalize averages and cease data collection.
        accumulate_result!(wta, model)

        # Averaging period is complete.
        wta.collecting = false

        # Reset the "previous" interval time,
        # subtracting a sliver that presents window overshoot from accumulating.
        wta.previous_interval_stop_time = model.clock.time - rem(model.clock.time, wta.time_interval)

    elseif mod(model.clock.iteration - wta.window_start_iteration, wta.stride) == 0
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
