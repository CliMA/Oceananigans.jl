using Dates: AbstractDateTime
using Oceananigans: AbstractModel, defaults, instantiated_location
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: AbstractField, Scan
using Oceananigans.Utils: time_difference_seconds

using Statistics: Statistics

import Oceananigans: initialize!, prognostic_state, restore_prognostic_state!
import Oceananigans.Grids: grid
import Oceananigans.Fields: location, indices, interior

"""
    mutable struct TimeDerivative{O, R, T}

Container that holds the state required to compute the time derivative of an `operand`
as a simulation runs: the `operand` recorded when the current differencing window opened at
`previous_time`, and the most recently completed `result`. Both are `Field`s at
`location(operand)`.
"""
mutable struct TimeDerivative{O, R, T}
           result :: R
          operand :: O
         previous :: R
    previous_time :: T
          pending :: Bool
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

An output writer holding a `TimeDerivative` actuates again on the iteration after each
output and writes the completed difference into the record it opened, so the record carries
the output time. A record the run ends before completing holds `NaN` (`NetCDFWriter` and
`ZarrWriter`) or is absent (`JLD2Writer`). To use a `TimeDerivative` without a writer,
construct a [`TimeDerivativeCallback`](@ref). Field operations are forwarded to `result`,
so `2 * ∂ₜc` builds the same `AbstractOperation` as `2 * ∂ₜc.result`.

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

    derivative = TimeDerivative(result, operand, previous, previous_time, false)

    isnothing(model) || initialize!(derivative, model)

    return derivative
end

similar_field(operand) = Field(instantiated_location(operand), operand.grid, eltype(operand),
                               indices = indices(operand))

grid(derivative::TimeDerivative) = grid(derivative.operand)
location(derivative::TimeDerivative) = location(derivative.operand)
indices(derivative::TimeDerivative) = indices(derivative.operand)

#####
##### Read a `TimeDerivative` like the `Field` it computes
#####

Base.parent(derivative::TimeDerivative) = parent(derivative.result)
Base.size(derivative::TimeDerivative, args...) = size(derivative.result, args...)
Base.eltype(derivative::TimeDerivative) = eltype(derivative.result)
Base.getindex(derivative::TimeDerivative, args...) = getindex(derivative.result, args...)

interior(derivative::TimeDerivative, args...) = interior(derivative.result, args...)

for reduction in (:sum, :maximum, :minimum, :all, :any, :prod, :extrema)
    @eval begin
        Base.$reduction(derivative::TimeDerivative; kw...) = Base.$reduction(derivative.result; kw...)
        Base.$reduction(f::Function, derivative::TimeDerivative; kw...) = Base.$reduction(f, derivative.result; kw...)
    end
end

Statistics.mean(derivative::TimeDerivative; kw...) = Statistics.mean(derivative.result; kw...)
Statistics.mean(f::Function, derivative::TimeDerivative; kw...) = Statistics.mean(f, derivative.result; kw...)

#####
##### Substitute `result` into the operators registered by `AbstractOperations`, so that
##### `2 * ∂ₜc` builds the same `AbstractOperation` as `2 * ∂ₜc.result`
#####

# Widening this union is ambiguous with the `op(::AbstractField, ::Any)` operator methods
const ScalarOperand = Union{Function, Number}

for op in (:sqrt, :sin, :cos, :exp, :tanh, :abs, :log10, :log, :tan, :sinh, :cosh, :-, :+)
    @eval Base.$op(derivative::TimeDerivative) = Base.$op(derivative.result)
end

for op in (:+, :-, :*, :/, :^, :>, :<, :>=, :<=, :atan, :atand, :mod)
    @eval begin
        Base.$op(a::TimeDerivative, b::ScalarOperand) = Base.$op(a.result, b)
        Base.$op(a::ScalarOperand, b::TimeDerivative) = Base.$op(a, b.result)
        Base.$op(a::TimeDerivative, b::TimeDerivative) = Base.$op(a.result, b.result)
    end
end

# Calling a `TimeDerivative` updates it; `fetch_output` reads it
fetch_output(derivative::TimeDerivative, model) = parent(derivative.result)

# The forward difference is only complete on the iteration after the writer actuates
deferred_output(::TimeDerivative) = true

(derivative::TimeDerivative)(sim) = update_time_derivative!(derivative, sim.model)

"""
$(TYPEDSIGNATURES)

Reset `derivative` so that its next actuation opens a fresh differencing window.
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

Complete the difference over the window opened at the previous actuation, storing it in
`derivative.result` labelled by `derivative.previous_time`, and reopen the window at the
current time. The first actuation only opens a window.
"""
function update_time_derivative!(derivative::TimeDerivative, model)
    current = fetch_output(derivative.operand, model)
    Δt = time_difference_seconds(model.clock.time, derivative.previous_time)

    if derivative.pending && Δt > 0
        # Difference over parents so that halo regions are included
        result = parent(derivative.result)
        previous = parent(derivative.previous)
        @. result = (current - previous) / Δt
    end

    # Every actuation opens a new window, so the next one differences back to this time
    if !derivative.pending || Δt > 0
        parent(derivative.previous) .= current
        derivative.previous_time = model.clock.time
        derivative.pending = true
    end

    return nothing
end

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
