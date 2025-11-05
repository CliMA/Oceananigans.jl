import Dates
using Dates: AbstractTime, Period, Nanosecond

# For Oceananigans clocks based on calendar values (e.g. DateTime or TimeDate),
# time-stepping in Oceananigans advances prognostic variables using a real-valued
# step measured in seconds. The clock converts back and forth by adding/subtracting
# `Dates.Nanosecond` offsets.  These helper functions centralise the
# conversions so that simulations, clocks, schedules, and I/O all agree on
# how numeric seconds and calendar periods map onto one another.

@inline seconds_to_nanosecond(seconds::Number) = Nanosecond(round(Int, seconds * 1e9))
@inline seconds_to_nanosecond(period::Period) = convert(Nanosecond, period)
@inline seconds_to_nanosecond(ns::Nanosecond) = ns

@inline period_to_seconds(seconds::Number) = float(seconds)
@inline period_to_seconds(period::Period) = Dates.value(convert(Nanosecond, period)) / 1e9

@inline function period_to_seconds(period, ::Type{T}) where T
    return convert(T, period_to_seconds(period))
end

@inline time_difference_seconds(target, current) = float(target - current)
@inline time_difference_seconds(target::AbstractTime, current::AbstractTime) = period_to_seconds(target - current)
@inline time_difference_seconds(target::Number, current::AbstractTime) = period_to_seconds(target - current)
@inline time_difference_seconds(target::AbstractTime, current::Number) = period_to_seconds(target - current)

@inline add_time_interval(base::Number, interval::Number, count=1) = base + count * interval
@inline add_time_interval(base::Number, interval::Period, count=1) = base + count * period_to_seconds(interval)
@inline add_time_interval(base::AbstractTime, interval::Number, count=1) = base + seconds_to_nanosecond(interval * count)
@inline add_time_interval(base::AbstractTime, interval::Period, count=1) = base + count * interval

@inline add_time_interval(base, interval::Array{<:Number}, count=1) = interval[count]
@inline add_time_interval(base, interval::Array{<:Dates.Period}, count=1) = seconds_to_nanosecond(interval[count])
@inline add_time_interval(base, interval::Array{Dates.DateTime}, count=1) = interval[count]

function period_type(interval::Number)
    FT = Oceananigans.defaults.FloatType
    return FT
end

function period_type(interval::Array{<:Number})
    FT = Oceananigans.defaults.FloatType
    return Array{FT, 1}
end

period_type(interval::Dates.Period) = typeof(interval)
period_type(interval) = typeof(interval)

time_type(interval::Number) = typeof(interval)
time_type(interval::Dates.Period) = Dates.DateTime
time_type(interval::Array) = eltype(interval)