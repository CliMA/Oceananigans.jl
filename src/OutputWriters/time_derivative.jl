using Adapt: Adapt
using Base: @propagate_inbounds
using Dates: AbstractDateTime
using Oceananigans: AbstractModel, defaults, instantiated_location
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: AbstractField, Scan
using Oceananigans.Utils: time_difference_seconds

import Oceananigans: initialize!, prognostic_state, restore_prognostic_state!
import Oceananigans.Fields: indices, interior

"""
    mutable struct TimeDerivative <: AbstractField

Container that holds the state required to compute the time derivative of an `operand`
as a simulation runs: the `operand` evaluated at the `previous_time`, and the most
recently computed `result`. Both are `Field`s at `location(operand)`, and the container
reads as `result`.
"""
mutable struct TimeDerivative{LX, LY, LZ, G, T, O, R, TT, FT} <: AbstractField{LX, LY, LZ, G, T, 3}
                           result :: R
                          operand :: O
                         previous :: R
                    previous_time :: TT
    expected_max_time_step_growth :: FT
                             grid :: G
end

materialize_operand(operand) = operand
materialize_operand(operand::Union{AbstractOperation, Scan}) = Field(operand)

"""
    TimeDerivative(operand, model=nothing; expected_max_time_step_growth=1.2)

Return an object that computes the time derivative of `operand` while a simulation runs,

```math
∂ₜ a ≈ \\frac{aⁿ - aⁿ⁻¹}{tⁿ - tⁿ⁻¹} \\, ,
```

where ``aⁿ`` and ``aⁿ⁻¹`` are `operand` evaluated at the two most recent times the derivative
was updated. The derivative is a backward difference, centered at ``tⁿ - Δt / 2`` with
``Δt = tⁿ - tⁿ⁻¹``, and is zero until `operand` has been evaluated twice.

`operand` may be a `Field`, an `AbstractOperation`, or a `Reduction`; operations and
reductions are materialized into a `Field` on construction. Δt is measured in seconds.

An output writer updates a `TimeDerivative` among its outputs through a
[`TimeDerivativeCallback`](@ref) that it registers itself; construct the callback directly
to use one without a writer. The writer's next actuation is anticipated by assuming that the
time step grows by at most a factor `expected_max_time_step_growth` in one iteration, as
described in [`PrecedingIterations`](@ref).

A `TimeDerivative` is an `AbstractField` that reads as its `result`, so it can be indexed,
reduced, and composed into operations like any other field. Reading it, including through
`compute!`, does not advance it: an operation built from a `TimeDerivative` sees whatever its
callback last computed, so keep a [`TimeDerivativeCallback`](@ref) in `simulation.callbacks`
when writing such operations.

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
                                                overwrite_files = true)

# output
JLD2Writer scheduled on TimeInterval(1 second):
├── filepath: tracer_variance_budget.jld2
├── 1 outputs: ∂ₜc²
├── array_type: Array{Float32}
├── including: [:coriolis, :buoyancy, :closure]
├── file_splitting: NoFileSplitting
└── file size: 0 bytes (file not yet created)
```
"""
function TimeDerivative(operand, model=nothing; expected_max_time_step_growth = 1.2)
    operand = materialize_operand(operand)

    result = similar_field(operand)
    previous = similar_field(operand)

    previous_time = isnothing(model) ? zero(defaults.FloatType) : model.clock.time
    grid = operand.grid
    LX, LY, LZ = location(operand)
    G, T, O, R = typeof(grid), eltype(operand), typeof(operand), typeof(result)
    TT, FT = typeof(previous_time), typeof(expected_max_time_step_growth)

    derivative = TimeDerivative{LX, LY, LZ, G, T, O, R, TT, FT}(result, operand, previous, previous_time,
                                                              expected_max_time_step_growth, grid)

    isnothing(model) || initialize!(derivative, model)

    return derivative
end

similar_field(operand) = Field(instantiated_location(operand), operand.grid, eltype(operand),
                               indices = indices(operand))

#####
##### Read a `TimeDerivative` like the `Field` it computes
#####

Base.parent(derivative::TimeDerivative) = parent(derivative.result)
Base.size(derivative::TimeDerivative) = size(derivative.result)

@propagate_inbounds Base.getindex(derivative::TimeDerivative, inds...) = getindex(derivative.result, inds...)

indices(derivative::TimeDerivative) = indices(derivative.result)
interior(derivative::TimeDerivative) = interior(derivative.result)
interior(derivative::TimeDerivative, I...) = interior(derivative.result, I...)

"Inside kernels a `TimeDerivative` is its `result`."
Adapt.adapt_structure(to, derivative::TimeDerivative) = Adapt.adapt(to, derivative.result)

# Calling a `TimeDerivative` updates it; reading it, including through `compute!`, does not
(derivative::TimeDerivative)(sim) = update_time_derivative!(derivative, sim.model)

"""
$(TYPEDSIGNATURES)

Record `derivative.operand` and the current time for the next update to difference against.
"""
function initialize!(derivative::TimeDerivative, model::AbstractModel)
    println("    [TimeDerivative] SEED   at iteration $(model.clock.iteration), t = $(model.clock.time), last_Δt = $(model.clock.last_Δt)")  # TEMPORARY
    if derivative.previous_time isa Number && model.clock.time isa AbstractDateTime
        T = typeof(model.clock.time)
        throw(ArgumentError("TimeDerivative must be constructed with the model when the clock keeps $T time"))
    end

    parent(derivative.previous) .= fetch_output(derivative.operand, model)
    derivative.previous_time = model.clock.time

    return nothing
end

initialize!(derivative::TimeDerivative, sim) = initialize!(derivative, sim.model)

"""
$(TYPEDSIGNATURES)

Difference `derivative.operand` against its value at `derivative.previous_time` and store
the result in `derivative.result`.
"""
function update_time_derivative!(derivative::TimeDerivative, model)
    println("    [TimeDerivative] UPDATE at iteration $(model.clock.iteration), t = $(model.clock.time), last_Δt = $(model.clock.last_Δt), differencing Δt = $(time_difference_seconds(model.clock.time, derivative.previous_time))")  # TEMPORARY
    Δt = time_difference_seconds(model.clock.time, derivative.previous_time)
    Δt == 0 && return nothing

    # Difference over parents so that halo regions are included
    current = fetch_output(derivative.operand, model)
    result = parent(derivative.result)
    previous = parent(derivative.previous)

    @. result = (current - previous) / Δt
    @. previous = current
    derivative.previous_time = model.clock.time

    return nothing
end

#####
##### Checkpointing
#####

function prognostic_state(derivative::TimeDerivative)
    return (result = prognostic_state(derivative.result),
            previous = prognostic_state(derivative.previous),
            previous_time = derivative.previous_time)
end

function restore_prognostic_state!(restored::TimeDerivative, from)
    restore_prognostic_state!(restored.result, from.result)
    restore_prognostic_state!(restored.previous, from.previous)
    restored.previous_time = from.previous_time
    return restored
end

restore_prognostic_state!(::TimeDerivative, ::Nothing) = nothing

#####
##### Show
#####

Base.summary(derivative::TimeDerivative) = string("TimeDerivative of ", summary(derivative.operand))

Base.show(io::IO, derivative::TimeDerivative) = print(io, summary(derivative))
