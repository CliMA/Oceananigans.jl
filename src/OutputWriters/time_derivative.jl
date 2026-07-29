using Dates: AbstractDateTime
using Oceananigans: defaults, instantiated_location
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: Scan
using Oceananigans.Utils: time_difference_seconds

import Oceananigans: initialize!, prognostic_state, restore_prognostic_state!
import Oceananigans.Grids: grid
import Oceananigans.Fields: location, indices

"""
    mutable struct TimeDerivative{O, R, T}

Container that holds the state required to compute the time derivative of an `operand`
as a simulation runs: the `operand` evaluated at the `previous_time`, and the most
recently computed `result`. Both are `Field`s at `location(operand)`.
"""
mutable struct TimeDerivative{O, R, T}
           result :: R
          operand :: O
         previous :: R
    previous_time :: T
end

materialize_operand(operand) = operand
materialize_operand(operand::Union{AbstractOperation, Scan}) = Field(operand)

"""
    TimeDerivative(operand, model=nothing)

Return an object that computes the time derivative of `operand` while a simulation runs,

```math
∂ₜ a ≈ \\frac{aⁿ - aⁿ⁻¹}{tⁿ - tⁿ⁻¹} \\, ,
```

where ``aⁿ`` and ``aⁿ⁻¹`` are `operand` evaluated at the two most recent times the derivative
was updated. The derivative is a backward difference and is therefore centered at
``tⁿ - Δt / 2``, where ``Δt = tⁿ - tⁿ⁻¹``. It is zero until `operand` has been evaluated
twice, which means that the derivative written at the start of a simulation is zero.

`operand` may be a `Field`, an `AbstractOperation`, or a `Reduction`; operations and
reductions are materialized into a `Field` on construction. Δt is measured in seconds,
including for clocks that keep calendar time.

A `TimeDerivative` is not an operator: it cannot be composed into further `AbstractOperation`s.
It is an output that is interpreted by an output writer, and appears in the `outputs` of a
writer like any other. Updating it is the job of a [`TimeDerivativeCallback`](@ref), which a
writer adds to `simulation.callbacks` on its own. Construct one directly to use a
`TimeDerivative` without a writer, or to difference over an interval longer than a time step.

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
JLD2Writer scheduled on TimeInterval(1 second):
├── filepath: tracer_variance_budget.jld2
├── 1 outputs: ∂ₜc²
├── array_type: Array{Float32}
├── including: [:coriolis, :buoyancy, :closure]
├── file_splitting: NoFileSplitting
└── file size: 0 bytes (file not yet created)
```

An output writer is not required. A [`TimeDerivativeCallback`](@ref) in
`simulation.callbacks` keeps a `TimeDerivative` up to date on its own schedule, and its
`result` is a `Field` that can be read at any point during the run:

```jldoctest time_derivative
∂ₜc = TimeDerivativeCallback(model.tracers.c, schedule=IterationInterval(1))

simulation.callbacks[:∂ₜc] = ∂ₜc

# output
Callback of TimeDerivative of 4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU on IterationInterval(1)
```
"""
function TimeDerivative(operand, model=nothing)
    operand = materialize_operand(operand)

    result = similar_field(operand)
    previous = similar_field(operand)

    previous_time = isnothing(model) ? zero(defaults.FloatType) : model.clock.time

    derivative = TimeDerivative(result, operand, previous, previous_time)

    isnothing(model) || seed_time_derivative!(derivative, model)

    return derivative
end

# Not `Base.similar`, which inherits `operand`'s `operand` and would recompute it
similar_field(operand) = Field(instantiated_location(operand), operand.grid, eltype(operand),
                               indices = indices(operand))

# Time derivatives don't change spatial location or grid
grid(derivative::TimeDerivative) = grid(derivative.operand)
location(derivative::TimeDerivative) = location(derivative.operand)
indices(derivative::TimeDerivative) = indices(derivative.operand)
Base.parent(derivative::TimeDerivative) = parent(derivative.result)

# Dispatched rather than left to the `output(model)` fallback, because calling a
# `TimeDerivative` updates it: that is what its `Callback` invokes every actuation.
fetch_output(derivative::TimeDerivative, model) = parent(derivative.result)

(derivative::TimeDerivative)(sim) = update_time_derivative!(derivative, sim.model)

"""
$(TYPEDSIGNATURES)

Record `derivative.operand` and the current time so that the next update has something to
difference against. A backward difference needs two evaluations, so the derivative remains
zero until the update after this one.
"""
function seed_time_derivative!(derivative::TimeDerivative, model)
    if derivative.previous_time isa Number && model.clock.time isa AbstractDateTime
        T = typeof(model.clock.time)
        msg = string("Cannot use a TimeDerivative with a $T clock unless it is constructed ",
                     "with the model, as in TimeDerivative(operand, model).")
        throw(ArgumentError(msg))
    end

    parent(derivative.previous) .= fetch_output(derivative.operand, model)
    derivative.previous_time = model.clock.time

    return nothing
end

initialize!(derivative::TimeDerivative, sim) = seed_time_derivative!(derivative, sim.model)

"""
$(TYPEDSIGNATURES)

Difference `derivative.operand` against its value at `derivative.previous_time` and store
the result in `derivative.result`.
"""
function update_time_derivative!(derivative::TimeDerivative, model)
    Δt = time_difference_seconds(model.clock.time, derivative.previous_time)
    Δt == 0 && return nothing

    # Broadcast over parents so that halo regions are differenced along with the interior,
    # matching what an output writer with `with_halos = true` saves for a plain `Field`
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
