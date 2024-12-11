using Printf
using Dates: AbstractTime

using Oceananigans.Units

maybe_int(t) = isinteger(t) ? Int(t) : t

"""
    prettytime(t, longform=true)

Convert a floating point value `t` representing an amount of time in
SI units of seconds to a human-friendly string with three decimal places.
Depending on the value of `t` the string will be formatted to show `t` in
nanoseconds (ns), microseconds (μs), milliseconds (ms),
seconds, minutes, hours, or days.

With `longform=false`, we use s, m, hrs, and d in place of seconds,
minutes, and hours.
"""
function prettytime(t, longform=true)
    # Modified from: https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/src/trials.jl
    value, units = prettytimeunits(t, longform)

    if iszero(value)
        msg = "0 $units"
    elseif value < 1e-9 
        msg = @sprintf("%.3e %s", value, units) # yah that's small
    elseif isinteger(value)
        msg = @sprintf("%d %s", value, units)
    else
        msg = @sprintf("%.3f %s", value, units)
    end

    return msg::String
end

function prettytimeunits(t::T, longform=true) where T
    if t < 1e-9 # just _forget_ picoseconds!
        value = t
        units = !longform ? "s" : "seconds"
    elseif t < 1e-6
        value = t * 1e9
        units = "ns"
    elseif t < 1e-3
        value = t * 1e6
        units = "μs"
    elseif t < 1
        value = t * 1e3
        units = "ms"
    elseif t < minute
        value = t
        units = !longform ? "s" : value==1 ? "second" : "seconds"
    elseif t < hour
        value = t / minute
        units = !longform ? "m" : value==1 ? "minute" : "minutes"
    elseif t < day
        value = t / hour
        units = !longform ? "hr" : value==1 ? "hour" : "hours"
    else
        value = t / day
        units = !longform ? "d" : value==1 ? "day" : "days"
    end

    return convert(T, value), units::String
end

prettytime(dt::AbstractTime, longform=true) = "$dt"
