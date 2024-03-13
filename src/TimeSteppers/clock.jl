using Adapt
using Dates: AbstractTime, DateTime, Nanosecond, Millisecond
using Oceananigans.Utils: prettytime

import Base: show
import Oceananigans.Units: Time

"""
    mutable struct Clock{T, FT}

Keeps track of the current `time`, `last_Δt`, `iteration` number, and time-stepping `stage`.
The `stage` is updated only for multi-stage time-stepping methods. The `time::T` is
either a number or a `DateTime` object.
"""
mutable struct Clock{T, FT}
         time :: T
      last_Δt :: FT
    iteration :: Int
        stage :: Int
end

"""
    Clock(; time, last_Δt = Inf, iteration=0, stage=1)

Returns a `Clock` object. By default, `Clock` is initialized to the zeroth `iteration`
and first time step `stage` with `last_Δt`.
"""
Clock(; time::T, last_Δt::FT = Inf, iteration=0, stage=1) where {T, FT} = Clock{T, FT}(time, last_Δt, iteration, stage)

Base.summary(clock::Clock) = string("Clock(time=$(prettytime(clock.time)), iteration=$(clock.iteration), last_Δt=$(prettytime(clock.last_Δt)))")

Base.show(io::IO, c::Clock{T}) where T =
    println(io, "Clock{$T}: time = $(prettytime(c.time)), last_Δt = $(prettytime(c.last_Δt)), iteration = $(c.iteration), stage = $(c.stage)")

next_time(clock, Δt) = clock.time + Δt
next_time(clock::Clock{<:AbstractTime}, Δt) = clock.time + Nanosecond(round(Int, 1e9 * Δt))

tick_time!(clock, Δt) = clock.time += Δt
tick_time!(clock::Clock{<:AbstractTime}, Δt) = clock.time += Nanosecond(round(Int, 1e9 * Δt))

Time(clock::Clock) = Time(clock.time)

# Convert the time to units of clock.time (assumed to be seconds if using DateTime or TimeDate).
unit_time(t) = t
unit_time(t::Millisecond) = t.value / 1_000
unit_time(t::Nanosecond) = t.value / 1_000_000_000

# Convert to a base Julia type (a float or DateTime). Mainly used by NetCDFOutputWriter.
float_or_date_time(t) = t
float_or_date_time(t::AbstractTime) = DateTime(t)

function tick!(clock, Δt; stage=false)

    tick_time!(clock, Δt)

    clock.last_Δt = Δt

    if stage # tick a stage update
        clock.stage += 1
    else # tick an iteration and reset stage
        clock.iteration += 1
        clock.stage = 1
    end

    return nothing
end

"Adapt `Clock` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, clock::Clock) = (time=clock.time, last_Δt=clock.last_Δt, iteration=clock.iteration, stage=clock.stage)
