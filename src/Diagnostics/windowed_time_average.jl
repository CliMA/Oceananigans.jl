import ..Utils: time_to_run
import ..Fields: Field

"""
    WindowedTimeAverage{RT, FT, A, B} <: Diagnostic

An object for computing 'windowed' time averages, or moving time-averages
of a `kernel` over a specified `time_window`, collected on `time_interval`.
"""
mutable struct WindowedTimeAverage{RT, FT, K, R} <: AbstractDiagnostic
                    time_window :: FT
                  time_interval :: FT
              window_start_time :: FT
       previous_collection_time :: FT
    previous_interval_stop_time :: FT
                     collecting :: Bool
                         kernel :: K
                         result :: R
                    return_type :: RT
end

"""
    WindowedTimeAverage(kernel; time_window, time_interval,
                                return_type=Array, float_type=Float64)
                                                        
Returns an object for computing running averages of `kernel` over `time_window`,
recurring on `time_interval`. Calling the `WindowedTimeAverage` object returns 
the computed time-average of `kernel` at the current time, converted to `return_type`.
`float_type` specifies the floating point precision of scalar parameters.

`kernel` may be an `Oceananigans.Field`, `Oceananigans.AbstractOperations.Computation,
or `Oceananigans.Diagnostics.Average`.
""" 
function WindowedTimeAverage(kernel; time_window, time_interval,
                                     return_type=Array, float_type=Float64)

    result = 0 .* deepcopy(get_kernel(kernel))

    return WindowedTimeAverage(float_type(time_window),
                               float_type(time_interval),
                               zero(float_type),
                               zero(float_type),
                               zero(float_type),
                               false,
                               kernel,
                               result,
                               return_type)
end

function time_to_run(clock, wta::WindowedTimeAverage)
    if (wta.collecting || 
        clock.time > wta.previous_interval_stop_time + wta.time_interval - wta.time_window)

        return true
    else
        return false
    end
end

get_kernel(field::Field) = parent(field)

function get_kernel(kernel::Average)
    run_diagnostic(nothing, kernel)
    return kernel.result
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
        wta.previous_collection_time = model.clock.time

    # We are currently collecting data:
    elseif model.clock.time - wta.window_start_time >= wta.time_window 
        # The averaging window is now closed. Finalize averages and cease data collection.

        # Accumulate final point in the left Riemann sum
        Δt = model.clock.time - wta.previous_collection_time
        wta.result .+= get_kernel(wta.kernel) .* Δt

        # Averaging period is complete.
        wta.collecting = false

        # Convert time integral to a time-average:
        wta.result ./= model.clock.time - wta.window_start_time

        # Reset the "previous" interval time,
        # subtracting a sliver that presents window overshoot from accumulating.
        wta.previous_interval_stop_time = model.clock.time - rem(model.clock.time, wta.time_interval)

    else # the averaging window is open; accumulate the time-average.

        # Accumulate left Riemann sum
        Δt = model.clock.time - wta.previous_collection_time
        wta.result .+= get_kernel(wta.kernel) .* Δt

        # Save data collection time
        wta.previous_collection_time = model.clock.time

    end

    return nothing
end

convert_result(wta) = wta.return_type(wta.result) # default
convert_result(wta::WindowedTimeAverage{<:Nothing}) = wta.result

function (wta::WindowedTimeAverage)(model=nothing)

    if wta.collecting 

        warn("The windowed time average is currently being collected.
             Converting intermediate result to a time average.")

        return convert_result(wta) ./ (wta.previous_collection_time - wta.window_start_time)
    else
        return convert_result(wta)
    end

end
