using Oceananigans: Oceananigans, prognostic_state, restore_prognostic_state!
using Oceananigans.Diagnostics: AbstractDiagnostic
using Oceananigans.Fields: Fields, indices, location
using Oceananigans.Grids: Grids, grid
using Oceananigans.OutputWriters: fetch_output, output_time, should_write_initial_output
using Oceananigans.Units: days, hours
using Oceananigans.Utils: AbstractSchedule, IterationInterval, prettytime

"""
    AbstractFilterKernel

Supertype for the weighting used by [`FilteredTimeInterval`](@ref). A kernel is called as
`kernel(τ, window)` and returns the weight of a sample at time offset `τ` from the center of
a frame, for `|τ| ≤ window / 2`.
"""
abstract type AbstractFilterKernel end

"""
$(TYPEDSIGNATURES)

Lanczos filter weights [Duchon (1979)](@cite duchon1979lanczos),

    w(τ) = sinc(2τ / cutoff) sinc(2τ / window),

which pass periods longer than `cutoff` and remove shorter ones.
"""
struct LanczosKernel{FT} <: AbstractFilterKernel
    cutoff :: FT
end

(kernel::LanczosKernel)(τ, window) = sinc(2τ / kernel.cutoff) * sinc(2τ / window)

Base.summary(kernel::LanczosKernel) = string("Lanczos, cutoff=", prettytime(kernel.cutoff))

"""
$(TYPEDSIGNATURES)

Uniform weights over the window, so each frame is a running mean.
"""
struct BoxcarKernel <: AbstractFilterKernel end

(::BoxcarKernel)(τ, window) = abs(τ) <= window / 2 ? one(τ) : zero(τ)

Base.summary(::BoxcarKernel) = "boxcar"

"""
$(TYPEDSIGNATURES)

Raised-cosine weights over the window,

    w(τ) = (1 + cos(2πτ / window)) / 2,    |τ| ≤ window / 2.
"""
struct HanningKernel <: AbstractFilterKernel end

(::HanningKernel)(τ, window) = abs(τ) <= window / 2 ? (1 + cos(2π * τ / window)) / 2 : zero(τ)

Base.summary(::HanningKernel) = "Hanning"

mutable struct FilteredTimeInterval{FT, K<:AbstractFilterKernel} <: AbstractSchedule
    interval :: FT
    window :: FT
    kernel :: K
    next_frame_number :: Int # frame k is centered at k * interval; 0 before the first time step
end

"""
$(TYPEDSIGNATURES)

Return a schedule for an output writer that writes its outputs filtered in time, one frame every
`interval`. Each frame is a weighted average of an output over a `window` centered on the frame time,
with weights given by `kernel`, an [`AbstractFilterKernel`](@ref): [`LanczosKernel`](@ref)`(cutoff)`,
[`HanningKernel`](@ref)`()` or [`BoxcarKernel`](@ref)`()`. `LanczosKernel` passes periods longer than `cutoff` and removes
shorter ones, so a cutoff of 40 hours removes the diurnal and semidiurnal tides.

Frames fall on multiples of `interval` and are stamped with their center time; each is written
`window / 2` after it. Frames whose window starts before the first time step are not written, so
a simulation picked up from a checkpoint writes its first frame `window / 2` after the checkpoint.

```jldoctest
using Oceananigans
using Oceananigans.Units

FilteredTimeInterval(LanczosKernel(40hours); interval=1days, window=5days)

# output
FilteredTimeInterval(interval=1 day, window=5 days, Lanczos, cutoff=1.667 days)
```
"""
function FilteredTimeInterval(kernel::AbstractFilterKernel; interval, window)
    interval, window = promote(interval, window)
    return FilteredTimeInterval(interval, window, kernel, 0)
end

Base.summary(filter::FilteredTimeInterval) = string("FilteredTimeInterval(interval=", prettytime(filter.interval),
                                             ", window=", prettytime(filter.window),
                                             ", ", summary(filter.kernel), ")")

Base.show(io::IO, filter::FilteredTimeInterval) = print(io, summary(filter))

(filter::FilteredTimeInterval)(model) = filter.next_frame_number > 0 &&
                                 model.clock.time >= filter.next_frame_number * filter.interval + filter.window / 2

# The frame that just became complete is written with its own center time rather than
# `clock.time` (when it fires, `window / 2` later). Called exactly once per write, since a
# writer's `schedule(model)` check (which doesn't call this) already gates whether
# `write_output!` — and so this — runs at all.
function output_time(clock, filter::FilteredTimeInterval)
    t = filter.next_frame_number * filter.interval
    filter.next_frame_number += 1
    return t
end

# A fresh simulation always writes every output writer once at iteration 0, regardless of what
# its schedule says at that point — `FilteredTimeInterval`'s first frame is never complete that early.
should_write_initial_output(::FilteredTimeInterval) = false

Oceananigans.prognostic_state(::FilteredTimeInterval) = nothing
Oceananigans.restore_prognostic_state!(::FilteredTimeInterval, ::Nothing) = nothing

mutable struct FilteredOutput{O, A, FT, K} <: AbstractDiagnostic
    operand :: O
    filter :: FilteredTimeInterval{FT, K}
    schedule :: IterationInterval
    sum_buffer :: Vector{A} # weighted sum of the operand, one per frame in progress
    weights :: Vector{FT}   # sum of the weights in each
    frames :: Vector{Int}   # frame each sum belongs to
    previous_time :: FT
end

function FilteredOutput(operand, filter, model)
    output = fetch_output(operand, model)
    Nbuffer = floor(Int, filter.window / filter.interval) + 1
    sum_buffer = [zero(output) for _ in 1:Nbuffer]
    FT = typeof(filter.interval)
    return FilteredOutput(operand, filter, IterationInterval(1), sum_buffer,
                                 zeros(FT, Nbuffer), zeros(Int, Nbuffer),
                                 convert(FT, model.clock.time))
end

function Oceananigans.run_diagnostic!(output::FilteredOutput, model)
    filter = output.filter
    t = model.clock.time
    Δt = t - output.previous_time
    output.previous_time = t

    # Frames whose window starts before the first time step seen are never complete.
    if filter.next_frame_number == 0
        filter.next_frame_number = ceil(Int, (t + filter.window / 2) / filter.interval)
    end

    φ = fetch_output(output.operand, model)
    first_frame = max(filter.next_frame_number, ceil(Int, (t - filter.window / 2) / filter.interval))
    last_frame = floor(Int, (t + filter.window / 2) / filter.interval)

    for frame in first_frame:last_frame
        n = mod(frame, length(output.sum_buffer)) + 1

        if output.frames[n] != frame
            output.frames[n] = frame
            output.sum_buffer[n] .= 0
            output.weights[n] = 0
        end

        τ = t - frame * filter.interval
        w = filter.kernel(τ, filter.window) * Δt
        output.sum_buffer[n] .+= w .* φ
        output.weights[n] += w
    end

    return nothing
end

function (output::FilteredOutput)(model)
    n = mod(output.filter.next_frame_number, length(output.sum_buffer)) + 1
    return output.sum_buffer[n] ./ output.weights[n]
end

Grids.grid(output::FilteredOutput) = grid(output.operand)
Fields.location(output::FilteredOutput) = location(output.operand)
Fields.indices(output::FilteredOutput) = indices(output.operand)

function time_average_outputs(filter::FilteredTimeInterval, outputs::NamedTuple, model)
    filtered_outputs = NamedTuple(name => FilteredOutput(outputs[name], filter, model) for name in keys(outputs))
    return filter, filtered_outputs
end

function time_average_outputs(filter::FilteredTimeInterval, outputs::AbstractDict, model)
    # `NetCDFWriter`/`ZarrWriter` pass an `OrderedDict`, not a `Dict`; build the same concrete
    # dictionary type back so their output order is preserved.
    DictType = Base.typename(typeof(outputs)).wrapper
    filtered_outputs = DictType(name => FilteredOutput(output, filter, model) for (name, output) in outputs)
    return filter, filtered_outputs
end
