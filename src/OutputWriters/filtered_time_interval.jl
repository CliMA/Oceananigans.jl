using Oceananigans: Oceananigans, prognostic_state, restore_prognostic_state!
using Oceananigans.Diagnostics: AbstractDiagnostic
using Oceananigans.Fields: Fields, indices, location
using Oceananigans.Grids: Grids, grid
using Oceananigans.OutputWriters: fetch_output, output_time, has_initial_output
using Oceananigans.Utils: AbstractSchedule, IterationInterval, prettytime

"""
    AbstractFilterKernel

Supertype for the weighting used by [`FilteredTimeInterval`](@ref). A kernel has a `window` field and
is called as `kernel(τ)`, returning the weight of a sample at time offset `τ` from the center of a
frame, for `|τ| ≤ window / 2`.
"""
abstract type AbstractFilterKernel end

"""
$(TYPEDSIGNATURES)

Lanczos filter weights [Duchon (1979)](@cite duchon1979lanczos) over `window`,

    w(τ) = sinc(2τ / cutoff) sinc(2τ / window),

which pass periods longer than `cutoff` and remove shorter ones.
"""
struct LanczosKernel{FT} <: AbstractFilterKernel
    window :: FT
    cutoff :: FT
end

function LanczosKernel(window; cutoff)
    window, cutoff = promote(window, cutoff)
    return LanczosKernel(window, cutoff)
end

(kernel::LanczosKernel)(τ) = sinc(2τ / kernel.cutoff) * sinc(2τ / kernel.window)

Base.summary(kernel::LanczosKernel) = string("LanczosKernel(window=", prettytime(kernel.window), ", cutoff=", prettytime(kernel.cutoff), ")")

"""
$(TYPEDSIGNATURES)

Uniform weights over `window`, so each frame is a running mean.
"""
struct BoxcarKernel{FT} <: AbstractFilterKernel
    window :: FT
end

(kernel::BoxcarKernel)(τ) = abs(τ) <= kernel.window / 2 ? one(τ) : zero(τ)

Base.summary(kernel::BoxcarKernel) = string("BoxcarKernel(window=", prettytime(kernel.window), ")")

"""
$(TYPEDSIGNATURES)

Raised-cosine weights over `window`,

    w(τ) = (1 + cos(2πτ / window)) / 2,    |τ| ≤ window / 2.
"""
struct HanningKernel{FT} <: AbstractFilterKernel
    window :: FT
end

(kernel::HanningKernel)(τ) = abs(τ) <= kernel.window / 2 ? (1 + cos(2π * τ / kernel.window)) / 2 : zero(τ)

Base.summary(kernel::HanningKernel) = string("HanningKernel(window=", prettytime(kernel.window), ")")

mutable struct FilteredTimeInterval{FT, K<:AbstractFilterKernel} <: AbstractSchedule
    interval :: FT
    kernel :: K
    next_frame_number :: Int # frame k is centered at k * interval; 0 before the first time step
end

"""
$(TYPEDSIGNATURES)

Return a schedule for an output writer that periodically writes time-filtered output on `interval`.
The output is a weighted average of an `AbstractField` (or a compatible function-based diagnostic)
over the `window` of `kernel`, centered on the output time. The weights come from `kernel`, an
[`AbstractFilterKernel`](@ref): [`LanczosKernel`](@ref)`(window; cutoff)`, [`HanningKernel`](@ref)`(window)`
or [`BoxcarKernel`](@ref)`(window)`. `LanczosKernel` passes periods longer than `cutoff` and removes
shorter ones, so a cutoff of 40 hours removes the diurnal and semidiurnal tides.

Outputs fall on multiples of `interval` and are stamped with their center time; each is written
`window / 2` after it. Outputs whose window starts before the first time step are not written, so
a simulation picked up from a checkpoint writes its first output `window / 2` after the checkpoint.

```jldoctest
using Oceananigans
using Oceananigans.Units

FilteredTimeInterval(LanczosKernel(5days; cutoff=40hours); interval=1days)

# output
FilteredTimeInterval(LanczosKernel(window=5 days, cutoff=1.667 days), interval=1 day)
```
"""
function FilteredTimeInterval(kernel::AbstractFilterKernel; interval)
    interval = convert(typeof(kernel.window), interval)
    return FilteredTimeInterval(interval, kernel, 0)
end

Base.show(io::IO, kernel::AbstractFilterKernel) = print(io, summary(kernel))

Base.summary(filter::FilteredTimeInterval) =
    string("FilteredTimeInterval(", summary(filter.kernel), ", interval=", prettytime(filter.interval), ")")

Base.show(io::IO, filter::FilteredTimeInterval) = print(io, summary(filter))

(filter::FilteredTimeInterval)(model) = filter.next_frame_number > 0 &&
                                 model.clock.time >= filter.next_frame_number * filter.interval + filter.kernel.window / 2

# The filtered output that just became complete is written with its own center time rather than
# `clock.time` (when it fires, `window / 2` later). Called exactly once per write, since a
# writer's `schedule(model)` check (which doesn't call this) already gates whether
# `write_output!` — and so this — runs at all.
function output_time(clock, filter::FilteredTimeInterval)
    t = filter.next_frame_number * filter.interval
    filter.next_frame_number += 1
    return t
end

# The first filtered output only exists once its window has been accumulated, so there is none at iteration 0.
has_initial_output(::FilteredTimeInterval) = false

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
    Nbuffer = floor(Int, filter.kernel.window / filter.interval) + 1
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
        filter.next_frame_number = ceil(Int, (t + filter.kernel.window / 2) / filter.interval)
    end

    φ = fetch_output(output.operand, model)
    first_frame = max(filter.next_frame_number, ceil(Int, (t - filter.kernel.window / 2) / filter.interval))
    last_frame = floor(Int, (t + filter.kernel.window / 2) / filter.interval)

    for frame in first_frame:last_frame
        n = mod(frame, length(output.sum_buffer)) + 1

        if output.frames[n] != frame
            output.frames[n] = frame
            output.sum_buffer[n] .= 0
            output.weights[n] = 0
        end

        τ = t - frame * filter.interval
        w = filter.kernel(τ) * Δt
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
