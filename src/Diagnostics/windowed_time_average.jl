import Oceananigans.Utils: time_to_run
import Oceananigans.Fields: AbstractField, compute!
import Oceananigans.OutputWriters: fetch_output

"""
    WindowedTimeAverage{RT, FT, A, B} <: Diagnostic

An object for computing 'windowed' time averages, or moving time-averages
of a `operand` over a specified `time_window`, collected on `time_interval`.
"""
mutable struct WindowedTimeAverage{RT, FT, O, R} <: AbstractDiagnostic
                              result :: R
                             operand :: O
                         time_window :: FT
                       time_interval :: FT
                              stride :: Int
                        field_slicer :: field_slicer
                   window_start_time :: FT
              window_start_iteration :: Int
            previous_collection_time :: FT
         previous_interval_stop_time :: FT
                          collecting :: Bool
end

"""
    WindowedTimeAverage(operand; time_window, time_interval, stride=1,
                                 float_type=Float64)
                                                        
Returns an object for computing running averages of `operand` over `time_window`,
recurring on `time_interval`. During the collection period, averages are computed
every `stride` iteration. 

`float_type` specifies the floating point precision of scalar parameters.

`operand` may be an `Oceananigans.Field`, `Oceananigans.AbstractOperations.Computation,
or `Oceananigans.Diagnostics.Average`.

Calling `wta(model)` for `wta::WindowedTimeAverage` object returns `wta.result`.
""" 
function WindowedTimeAverage(operand, model=nothing; time_window, time_interval, stride=1,
                                                     field_slicer = FieldSlicer(),
                                                     float_type=Float64)

    result = 0 .* deepcopy(fetch_output(operand, model, field_slicer))

    return WindowedTimeAverage(
                               result,
                               operand,
                               float_type(time_window),
                               float_type(time_interval),
                               stride,
                               field_slicer,
                               zero(float_type),
                               0,
                               zero(float_type),
                               zero(float_type),
                               false
                              )
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
    T_previous = wta.previous_collection_time - wta.window_start_time
    T_current = model.clock.time - wta.window_start_time

    # Accumulate left Riemann sum
    integrand = fetch_output(wta.operand, model, wta.field_slicer)
    @. wta.result = (wta.result * T_previous + integrand * Δt) / T_current
                    
    return nothing
end

function run_diagnostic(model, wta::WindowedTimeAverage)

    if !(wta.collecting)
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

        # Save data collection time
        wta.previous_collection_time = model.clock.time
    end

    return nothing
end

function (wta::WindowedTimeAverage)(model=nothing)
    wta.collecting && @warn "Returning a WindowedTimeAverage before the collection period is complete."
    return wta.result
end
