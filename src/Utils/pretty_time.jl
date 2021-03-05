using Printf
using Dates: AbstractTime

using Oceananigans.Units

maybe_int(t) = isinteger(t) ? Int(t) : t

"""
    prettytime(t)

Convert a floating point value `t` representing an amount of time in seconds to a more
human-friendly formatted string with three decimal places. Depending on the value of `t`
the string will be formatted to show `t` in nanoseconds (ns), microseconds (μs),
milliseconds (ms), seconds, minutes, hours, days, or years.
"""
function prettytime(t)
    # Modified from: https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/src/trials.jl
    iszero(t) && return "0 seconds"

    t = maybe_int(t)

    if t < 1e-9
        # No point going to picoseconds, just return something readable.
        return @sprintf("%.3e seconds", t)
    elseif t < 1e-6
        value, units = t * 1e9, "ns"
    elseif t < 1e-3
        value, units = t * 1e6, "μs"
    elseif t < 1
        value, units = t * 1e3, "ms"
    elseif t < minute
        value = t
        units = value == 1 ? "second" : "seconds"
    elseif t < hour
        value = maybe_int(t / minute)
        units = value == 1 ? "minute" : "minutes"
    elseif t < day
        value = maybe_int(t / hour)
        units = value == 1 ? "hour" : "hours"
    elseif t < year
        value = maybe_int(t / day)
        units = value == 1 ? "day" : "days"
    else
        value = maybe_int(t / year)
        units = value == 1 ? "year" : "years"
    end

    if isinteger(value)
        return @sprintf("%d %s", value, units)
    else
        return @sprintf("%.3f %s", value, units)
    end
end

prettytime(dt::AbstractTime) = "$dt"
