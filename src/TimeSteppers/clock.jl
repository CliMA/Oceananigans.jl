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
mutable struct Clock{TT, DT}
    time :: TT
    last_Δt :: DT
    last_stage_Δt :: DT
    iteration :: Int
    stage :: Int
end

"""
    Clock(; time, last_Δt=Inf, last_stage_Δt=Inf, iteration=0, stage=1)

Returns a `Clock` object. By default, `Clock` is initialized to the zeroth `iteration`
and first time step `stage` with `last_Δt=last_stage_Δt=Inf`.
"""
function Clock(; time,
               last_Δt = Inf,
               last_stage_Δt = Inf,
               iteration = 0,
               stage = 1)

    TT = typeof(time)
    DT = typeof(last_Δt)
    last_stage_Δt = convert(DT, last_Δt)
    return Clock{TT, DT}(time, last_Δt, last_stage_Δt, iteration, stage)
end

# TODO: when supporting DateTime, this function will have to be extended
time_step_type(TT) = TT

function Clock{TT}(; time,
                   last_Δt = Inf,
                   last_stage_Δt = Inf,
                   iteration = 0,
                   stage = 1) where TT

    DT = time_step_type(TT)
    last_Δt = convert(DT, last_Δt)
    last_stage_Δt = convert(DT, last_stage_Δt)

    return Clock{TT, DT}(time, last_Δt, last_stage_Δt, iteration, stage)
end

function Base.summary(clock::Clock)
    TT = typeof(clock.time)
    DT = typeof(clock.last_Δt)
    return string("Clock{", TT, ", ", DT, "}",
                  "(time=", prettytime(clock.time),
                  " iteration=", clock.iteration,
                  " last_Δt=", prettytime(clock.last_Δt), ")")
end

function Base.show(io::IO, clock::Clock)
    return print(io, summary(clock), '\n',
                 "├── stage: ", clock.stage, '\n',
                 "└── last_stage_Δt: ", prettytime(clock.last_stage_Δt))
end

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

    if stage # tick a stage update
        clock.stage += 1
    else # tick an iteration and reset stage
        clock.iteration += 1
        clock.stage = 1
    end

    return nothing
end

"""Adapt `Clock` for GPU."""
Adapt.adapt_structure(to, clock::Clock) = (time          = clock.time,
                                           last_Δt       = clock.last_Δt,
                                           last_stage_Δt = clock.last_stage_Δt,
                                           iteration     = clock.iteration,
                                           stage         = clock.stage)
    

