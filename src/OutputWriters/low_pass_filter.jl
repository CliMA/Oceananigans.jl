using Oceananigans: Oceananigans, prognostic_state, restore_prognostic_state!
using Oceananigans.Diagnostics: AbstractDiagnostic
using Oceananigans.Fields: Fields, indices, location
using Oceananigans.Grids: Grids, grid
using Oceananigans.OutputWriters: fetch_output, output_time, should_write_initial_output
using Oceananigans.Units: days, hours
using Oceananigans.Utils: AbstractSchedule, IterationInterval, prettytime

mutable struct LowPassFilter{FT} <: AbstractSchedule
    interval :: FT
    window :: FT
    cutoff :: FT
    next_frame :: Int # frame k is centered at k * interval; 0 before the first time step
end

"""
$(TYPEDSIGNATURES)

Return a schedule for an output writer that writes its outputs low-pass filtered in time, one
frame every `interval`. Each frame averages an output over a `window` centered on the frame time
with the weights of a Lanczos filter [Duchon (1979)](@cite duchon1979lanczos),

    w(τ) = sinc(2τ / cutoff) sinc(2τ / window),    |τ| ≤ window / 2,

which pass periods longer than `cutoff` and remove shorter ones. The defaults remove the diurnal
and semidiurnal tides.

Frames fall on multiples of `interval` and are stamped with their center time; each is written
`window / 2` after it. Frames whose window starts before the first time step are not written, so
a simulation picked up from a checkpoint writes its first frame `window / 2` after the checkpoint.

```jldoctest
using Oceananigans
using Oceananigans.Units

LowPassFilter(1days)

# output
LowPassFilter(interval=1 day, window=5 days, cutoff=1.667 days)
```
"""
function LowPassFilter(interval; window = 5days, cutoff = 40hours)
    interval, window, cutoff = promote(interval, window, cutoff)
    return LowPassFilter(interval, window, cutoff, 0)
end

Base.summary(filter::LowPassFilter) = string("LowPassFilter(interval=", prettytime(filter.interval),
                                             ", window=", prettytime(filter.window),
                                             ", cutoff=", prettytime(filter.cutoff), ")")

Base.show(io::IO, filter::LowPassFilter) = print(io, summary(filter))

(filter::LowPassFilter)(model) = filter.next_frame > 0 &&
                                 model.clock.time >= filter.next_frame * filter.interval + filter.window / 2

# The frame that just became complete is written with its own center time rather than
# `clock.time` (when it fires, `window / 2` later). Called exactly once per write, since a
# writer's `schedule(model)` check (which doesn't call this) already gates whether
# `write_output!` — and so this — runs at all.
function output_time(clock, filter::LowPassFilter)
    t = filter.next_frame * filter.interval
    filter.next_frame += 1
    return t
end

# A fresh simulation always writes every output writer once at iteration 0, regardless of what
# its schedule says at that point — `LowPassFilter`'s first frame is never complete that early.
should_write_initial_output(::LowPassFilter) = false

Oceananigans.prognostic_state(::LowPassFilter) = nothing
Oceananigans.restore_prognostic_state!(::LowPassFilter, ::Nothing) = nothing

mutable struct LowPassFilteredOutput{O, A, FT} <: AbstractDiagnostic
    operand :: O
    filter :: LowPassFilter{FT}
    schedule :: IterationInterval
    sums :: Vector{A}     # weighted sum of the operand, one per frame in progress
    weights :: Vector{FT} # sum of the weights in each
    frames :: Vector{Int} # frame each sum belongs to
    previous_time :: FT
end

function LowPassFilteredOutput(operand, filter, model)
    output = fetch_output(operand, model)
    frames_in_progress = floor(Int, filter.window / filter.interval) + 1
    sums = [zero(output) for _ in 1:frames_in_progress]
    FT = typeof(filter.interval)
    return LowPassFilteredOutput(operand, filter, IterationInterval(1), sums,
                                 zeros(FT, frames_in_progress), zeros(Int, frames_in_progress),
                                 convert(FT, model.clock.time))
end

function Oceananigans.run_diagnostic!(output::LowPassFilteredOutput, model)
    filter = output.filter
    t = model.clock.time
    Δt = t - output.previous_time
    output.previous_time = t

    # Frames whose window starts before the first time step seen are never complete.
    if filter.next_frame == 0
        filter.next_frame = ceil(Int, (t + filter.window / 2) / filter.interval)
    end

    φ = fetch_output(output.operand, model)
    first_frame = max(filter.next_frame, ceil(Int, (t - filter.window / 2) / filter.interval))
    last_frame = floor(Int, (t + filter.window / 2) / filter.interval)

    for frame in first_frame:last_frame
        n = mod(frame, length(output.sums)) + 1

        if output.frames[n] != frame
            output.frames[n] = frame
            output.sums[n] .= 0
            output.weights[n] = 0
        end

        τ = t - frame * filter.interval
        w = sinc(2τ / filter.cutoff) * sinc(2τ / filter.window) * Δt
        output.sums[n] .+= w .* φ
        output.weights[n] += w
    end

    return nothing
end

function (output::LowPassFilteredOutput)(model)
    n = mod(output.filter.next_frame, length(output.sums)) + 1
    return output.sums[n] ./ output.weights[n]
end

Grids.grid(output::LowPassFilteredOutput) = grid(output.operand)
Fields.location(output::LowPassFilteredOutput) = location(output.operand)
Fields.indices(output::LowPassFilteredOutput) = indices(output.operand)

function time_average_outputs(filter::LowPassFilter, outputs::NamedTuple, model)
    filtered_outputs = NamedTuple(name => LowPassFilteredOutput(outputs[name], filter, model) for name in keys(outputs))
    return filter, filtered_outputs
end

function time_average_outputs(filter::LowPassFilter, outputs::AbstractDict, model)
    # `NetCDFWriter`/`ZarrWriter` pass an `OrderedDict`, not a `Dict`; build the same concrete
    # dictionary type back so their output order is preserved.
    DictType = Base.typename(typeof(outputs)).wrapper
    filtered_outputs = DictType(name => LowPassFilteredOutput(output, filter, model) for (name, output) in outputs)
    return filter, filtered_outputs
end
