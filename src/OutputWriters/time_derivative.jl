using Adapt: Adapt
using Dates: AbstractDateTime
using Oceananigans: AbstractModel, defaults, instantiated_location
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: AbstractField, Field, Scan, location
using Oceananigans.Utils: time_difference_seconds

import Oceananigans: initialize!, prognostic_state, restore_prognostic_state!
import Oceananigans.Fields: compute_at!, compute!, indices, interior

"""
    mutable struct TimeDerivative <: AbstractField

Container that holds the state required to compute the time derivative of an `operand`
as a simulation runs: the `operand` recorded when the current differencing window opened at
`previous_time`, and the most recently completed `result`. Both are `Field`s at
`location(operand)`.
"""
mutable struct TimeDerivative{LX, LY, LZ, G, T, O, R, TT} <: AbstractField{LX, LY, LZ, G, T, 3}
           result :: R
          operand :: O
         previous :: R
    previous_time :: TT
          pending :: Bool
             grid :: G
end

materialize_operand(operand) = operand
materialize_operand(operand::Union{AbstractOperation, Scan}) = Field(operand)

"""
    TimeDerivative(operand, model=nothing)

Return an object that computes the time derivative of `operand` while a simulation runs,

```math
∂ₜ a \\, (tⁿ) ≈ \\frac{aⁿ⁺¹ - aⁿ}{tⁿ⁺¹ - tⁿ} \\, ,
```

a forward difference labelled by the time ``tⁿ`` at which its window opens and completed at
the following actuation ``tⁿ⁺¹``.

`operand` may be a `Field`, an `AbstractOperation`, or a `Reduction`; operations and
reductions are materialized into a `Field` on construction. Δt is measured in seconds.

A `TimeDerivative` is an `AbstractField`, so it composes into further operations and
reductions, and evaluating it — computing it, or computing any output built from it —
advances it. An output writer holding a `TimeDerivative`, or any output containing one,
actuates again on the iteration after each output and writes the completed difference into
the record it opened, so the record carries the output time. A record the run ends before
completing holds `NaN` (`NetCDFWriter` and `ZarrWriter`) or is absent (`JLD2Writer`). To
use a `TimeDerivative` without a writer, construct a [`TimeDerivativeCallback`](@ref).

Because a `TimeDerivative` holds a single differencing window, it should be evaluated on
one cadence: sharing one between writers with different schedules corrupts the interval
its differences span.

Example
=======

Closing a tracer variance budget online rather than differencing two snapshots offline:

```jldoctest time_derivative
using Oceananigans

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))

model = NonhydrostaticModel(grid, tracers=:c)

∂ₜc² = TimeDerivative(Integral(model.tracers.c^2))

# output
TimeDerivative of 1×1×1 Field{Nothing, Nothing, Nothing} reduced over dims = (1, 2, 3) on RectilinearGrid on CPU
```

The derivative is written like any other output:

```jldoctest time_derivative
simulation = Simulation(model, Δt=1, stop_iteration=10)

simulation.output_writers[:budget] = JLD2Writer(model, (; ∂ₜc²),
                                                filename = "tracer_variance_budget.jld2",
                                                schedule = TimeInterval(1),
                                                overwrite_existing = true)

# output
JLD2Writer scheduled on ConsecutiveIterations(TimeInterval(1 second), 1):
├── filepath: tracer_variance_budget.jld2
├── 1 outputs: ∂ₜc²
├── array_type: Array{Float32}
├── including: [:coriolis, :buoyancy, :closure]
├── file_splitting: NoFileSplitting
└── file size: 0 bytes (file not yet created)
```
"""
function TimeDerivative(operand, model=nothing)
    operand = materialize_operand(operand)

    result = similar_field(operand)
    previous = similar_field(operand)

    previous_time = isnothing(model) ? zero(defaults.FloatType) : model.clock.time
    grid = operand.grid
    LX, LY, LZ = location(operand)

    derivative = TimeDerivative{LX, LY, LZ, typeof(grid), eltype(operand), typeof(operand),
                                typeof(result), typeof(previous_time)}(result, operand, previous,
                                                                       previous_time, false, grid)

    isnothing(model) || initialize!(derivative, model)

    return derivative
end

similar_field(operand) = Field(instantiated_location(operand), operand.grid, eltype(operand),
                               indices = indices(operand))

#####
##### Read a `TimeDerivative` like the `Field` it computes
#####

Base.parent(derivative::TimeDerivative) = parent(derivative.result)
Base.size(derivative::TimeDerivative, args...) = size(derivative.result, args...)
Base.getindex(derivative::TimeDerivative, args...) = getindex(derivative.result, args...)

indices(derivative::TimeDerivative) = indices(derivative.result)
interior(derivative::TimeDerivative, args...) = interior(derivative.result, args...)

"Inside kernels a `TimeDerivative` is its `result`."
Adapt.adapt_structure(to, derivative::TimeDerivative) = Adapt.adapt(to, derivative.result)

# The forward difference is only complete on the iteration after the writer actuates
deferred_output(::TimeDerivative) = true

(derivative::TimeDerivative)(sim) = update_time_derivative!(derivative, sim.model.clock.time)

compute_at!(derivative::TimeDerivative, t) = (update_time_derivative!(derivative, t); derivative)
compute!(derivative::TimeDerivative, time=nothing) = compute_at!(derivative, time)

"""
$(TYPEDSIGNATURES)

Reset `derivative` so that its next evaluation opens a fresh differencing window.
"""
function initialize!(derivative::TimeDerivative, model::AbstractModel)
    if derivative.previous_time isa Number && model.clock.time isa AbstractDateTime
        T = typeof(model.clock.time)
        throw(ArgumentError("TimeDerivative must be constructed with the model when the clock keeps $T time"))
    end

    derivative.pending = false

    return nothing
end

initialize!(derivative::TimeDerivative, sim) = initialize!(derivative, sim.model)

"""
$(TYPEDSIGNATURES)

Complete the difference over the window opened at the previous evaluation, storing it in
`derivative.result` labelled by `derivative.previous_time`, and reopen the window at `t`.
The first evaluation only opens a window, and repeated evaluation at one time is a no-op.
"""
function update_time_derivative!(derivative::TimeDerivative, t)
    Δt = time_difference_seconds(t, derivative.previous_time)
    derivative.pending && Δt <= 0 && return nothing

    compute_at!(derivative.operand, t)
    current = parent(derivative.operand)

    if derivative.pending
        # Difference over parents so that halo regions are included
        result = parent(derivative.result)
        previous = parent(derivative.previous)
        @. result = (current - previous) / Δt
    end

    parent(derivative.previous) .= current
    derivative.previous_time = t
    derivative.pending = true

    return nothing
end

update_time_derivative!(::TimeDerivative, ::Nothing) = nothing

#####
##### Checkpointing
#####

function prognostic_state(derivative::TimeDerivative)
    return (result = prognostic_state(derivative.result),
            previous = prognostic_state(derivative.previous),
            previous_time = derivative.previous_time,
            pending = derivative.pending)
end

function restore_prognostic_state!(restored::TimeDerivative, from)
    restore_prognostic_state!(restored.result, from.result)
    restore_prognostic_state!(restored.previous, from.previous)
    restored.previous_time = from.previous_time
    restored.pending = from.pending
    return restored
end

restore_prognostic_state!(::TimeDerivative, ::Nothing) = nothing

#####
##### Show
#####

Base.summary(derivative::TimeDerivative) = string("TimeDerivative of ", summary(derivative.operand))

Base.show(io::IO, derivative::TimeDerivative) = print(io, summary(derivative))
Base.show(io::IO, ::MIME"text/plain", derivative::TimeDerivative) = print(io, summary(derivative))
