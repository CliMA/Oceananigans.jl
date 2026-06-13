using Adapt: Adapt
using Dates: AbstractTime, Nanosecond, Millisecond
using DocStringExtensions: TYPEDEF, TYPEDSIGNATURES
using Oceananigans.Utils: prettytime, seconds_to_nanosecond
using Oceananigans.Grids: AbstractGrid

import Oceananigans: restore_prognostic_state!
import Oceananigans.Units: Time
import Oceananigans.Fields: set!

"""
$(TYPEDEF)

Keeps track of the current `time`, `last_Δt`, `iteration` number, and time-stepping `stage`.
The `stage` is updated only for multi-stage time-stepping methods. The `time :: TT` is
either a `Number` or a `DateTime` object. `KT` is the floating point type used for
`time` when `clock` is passed to kernels.
"""
mutable struct Clock{TT, KT, DT, IT, S}
    time :: TT
    last_Δt :: DT
    last_stage_Δt :: DT
    iteration :: IT
    stage :: S
end

Clock{TT, DT, IT, S}(time, last_Δt, last_stage_Δt, iteration, stage) where {TT, DT, IT, S} =
    Clock{TT, TT, DT, IT, S}(time, last_Δt, last_stage_Δt, iteration, stage)

"""
    Clock(; time, last_Δt=Inf, last_stage_Δt=Inf, iteration=0, stage=1)

Return a `Clock` object. By default, `Clock` is initialized to the zeroth `iteration`
and first time step `stage` with `last_Δt=last_stage_Δt=Inf`.
"""
function Clock(; time,
               last_Δt = Inf,
               last_stage_Δt = Inf,
               iteration = 0,
               stage = 1,
               kernel_time_type = typeof(time))

    TT = typeof(time)
    DT = typeof(last_Δt)
    IT = typeof(iteration)
    last_stage_Δt = convert(DT, last_stage_Δt)
    return Clock{TT, kernel_time_type, DT, IT, typeof(stage)}(time, last_Δt, last_stage_Δt, iteration, stage)
end

materialize_clock!(clock::Clock, timestepper) = nothing

function reset!(clock::Clock{TT, KT, DT, IT, S}) where {TT, KT, DT, IT, S}
    clock.time = zero(TT)
    clock.iteration = zero(IT)
    clock.stage = 1
    clock.last_Δt = Inf
    clock.last_stage_Δt = Inf
    return nothing
end

"""
$(TYPEDSIGNATURES)

Set `clock` to the `new_clock`.
"""
function set!(clock::Clock, new_clock::Clock)
    clock.time = new_clock.time
    clock.iteration = new_clock.iteration
    clock.last_Δt = new_clock.last_Δt
    clock.last_stage_Δt = new_clock.last_stage_Δt
    clock.stage = new_clock.stage

    return nothing
end

function Base.:(==)(clock1::Clock, clock2::Clock)
    return clock1.time == clock2.time &&
           clock1.iteration == clock2.iteration &&
           clock1.last_Δt == clock2.last_Δt &&
           clock1.last_stage_Δt == clock2.last_stage_Δt &&
           clock1.stage == clock2.stage
end

function Base.isapprox(a::Clock, b::Clock; kw...)
    return isapprox(a.time, b.time; kw...) &&
           isapprox(a.last_Δt, b.last_Δt; kw...) &&
           isapprox(a.last_stage_Δt, b.last_stage_Δt; kw...) &&
           a.iteration == b.iteration &&
           a.stage == b.stage
end

# Type used to represent the time step Δt for a clock with `time::TT`.
# For numeric clocks, Δt has the same type as `time`. For `DateTime`/`AbstractTime`
# clocks, Δt is a `Float64` (interpreted as seconds).
time_step_type(TT) = TT
time_step_type(::Type{<:AbstractTime}) = Float64

function Clock{TT}(; time,
                   last_Δt = Inf,
                   last_stage_Δt = Inf,
                   iteration = 0,
                   stage = 1,
                   kernel_time_type = TT) where TT

    DT = time_step_type(TT)
    last_Δt = convert(DT, last_Δt)
    last_stage_Δt = convert(DT, last_stage_Δt)
    IT = typeof(iteration)

    return Clock{TT, kernel_time_type, DT, IT, typeof(stage)}(time, last_Δt, last_stage_Δt, iteration, stage)
end

# helpful default
Clock(grid::AbstractGrid{FT}) where {FT} = Clock{Float64}(; time=0, kernel_time_type=FT)

kernel_time_type(::Clock{TT, KT, DT, IT, S}) where {TT, KT, DT, IT, S} = KT

function Base.summary(clock::Clock)
    TT = typeof(clock.time)
    DT = typeof(clock.last_Δt)
    return string("Clock{", TT, ", ", DT, "}",
                  "(time=", prettytime(clock.time),
                  ", iteration=", clock.iteration,
                  ", last_Δt=", prettytime(clock.last_Δt), ")")
end

function Base.show(io::IO, clock::Clock)
    return print(io, summary(clock), '\n',
                 "├── stage: ", clock.stage, '\n',
                 "└── last_stage_Δt: ", prettytime(clock.last_stage_Δt))
end

next_time(clock, Δt) = clock.time + Δt
next_time(clock::Clock{<:AbstractTime}, Δt) = clock.time + seconds_to_nanosecond(Δt)

tick_time!(clock, Δt) = clock.time += Δt
tick_time!(clock::Clock{<:AbstractTime}, Δt) = clock.time += seconds_to_nanosecond(Δt)

Time(clock::Clock) = Time(clock.time)

# Convert the time to units of clock.time (assumed to be seconds if using DateTime or TimeDate).
unit_time(t) = t
unit_time(t::Millisecond) = t.value / 1_000
unit_time(t::Nanosecond) = t.value / 1_000_000_000

# Advance clock by a full time step Δt. Increments iteration and resets stage.
function tick!(clock, Δt)
    tick_time!(clock, Δt)
    clock.iteration += 1
    clock.stage = 1
    clock.last_Δt = Δt
    clock.last_stage_Δt = Δt
    return nothing
end

# Advance clock by stage_Δt for an intermediate stage. Increments stage counter.
function tick_stage!(clock, stage_Δt)
    tick_time!(clock, stage_Δt)
    clock.stage += 1
    clock.last_stage_Δt = stage_Δt
    return nothing
end

# Advance clock by stage_Δt for the final stage of a multi-stage method.
# Records step_Δt as last_Δt (the full time step size). Increments iteration and resets stage.
function tick_stage!(clock, stage_Δt, step_Δt)
    tick_time!(clock, stage_Δt)
    clock.iteration += 1
    clock.stage = 1
    clock.last_Δt = step_Δt
    clock.last_stage_Δt = stage_Δt
    return nothing
end

"""
$(TYPEDSIGNATURES)

Adapt `Clock` to an immutable kernel argument.
"""
function Adapt.adapt_structure(to, clock::Clock)
    KT = kernel_time_type(clock)
    DT = time_step_type(KT)

    return (time          = convert(KT, clock.time),
            last_Δt       = convert(DT, clock.last_Δt),
            last_stage_Δt = convert(DT, clock.last_stage_Δt),
            iteration     = clock.iteration,
            stage         = clock.stage)
end

"""
$(TYPEDSIGNATURES)

Return an immutable clock-like object with `time` demoted to `eltype(grid)`, all other fields unchanged.
Call this at every `launch!` site that passes `clock` to a kernel so that `clock.time`
inside the kernel matches grid precision rather than the clock's native Float64.
"""
function convert_time(grid, clock::Clock)
    FT = eltype(grid)
    kernel_clock = Clock(; time          = clock.time,
                           last_Δt       = clock.last_Δt,
                           last_stage_Δt = clock.last_stage_Δt,
                           iteration     = clock.iteration,
                           stage         = clock.stage,
                           kernel_time_type = FT)
    return Adapt.adapt(nothing, kernel_clock)
end


"""Restore the clock from a checkpointed state."""
function restore_prognostic_state!(restored::Clock, from)
    restored.time = from.time
    restored.iteration = from.iteration
    restored.last_Δt = from.last_Δt
    restored.last_stage_Δt = from.last_stage_Δt
    restored.stage = from.stage
    return restored
end

restore_prognostic_state!(::Clock, ::Nothing) = nothing
