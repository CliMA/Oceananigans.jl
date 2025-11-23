# Model interface

Oceananigans models are concrete subtypes of `AbstractModel` that can be advanced
by [`Simulation`](@ref Simulation). This page documents the *model interface*:
the handful of functions and data that a model must provide so that Simulation
can initialize it, march it forward, and coordinate callbacks, diagnostics, and
output writers.

The interface lives primarily inside `src/Simulations/` and is intentionally
minimal so that new models (or pedagogical toy models) can be written without
depending on the full ocean model infrastructure.

## Lifecycle overview

When `run!(sim::Simulation)` is called the following high-level sequence occurs:

1. `initialize!(sim.model)` prepares the model state (allocations, halo fills,
   etc.) and `update_state!(sim.model)` computes any auxiliary tendencies.
2. For every time step, Simulation computes an aligned `Δt`, gathers callbacks
   that should run inside the model (`ModelCallsite`s), and calls
   `time_step!(sim.model, Δt; callbacks=model_callbacks)`.
3. After the model finishes its step, Simulation executes diagnostics, output
   writers, and callbacks scheduled on the `TimeStepCallsite`.

Because `Simulation` assumes this protocol, any custom `AbstractModel` must
implement (or inherit sane fallbacks for) the items listed below.

## Functions invoked by `Simulation`

### Metadata and bookkeeping

- `model.clock :: Clock`: the source of truth for `time(model)` and
  `iteration(model)`. Simulation uses it for stop criteria and logging, and
  resets it via `reset_clock!(model)` when `reset!(sim)` is called.

- `model.grid`: must support `architecture(model) = model.grid.architecture`
  and `eltype(model) = eltype(model.grid)`. These functions drive how Δt is
  validated (`validate_Δt`) and how numbers are converted inside Simulation.

- `timestepper(model)`: Simulation resets the timestepper before the first time
  step by calling `reset!(timestepper(model))`. Lightweight models can store
  `timestepper = nothing` and rely on the `reset!(::Nothing) = nothing`
  fallback, but the accessor still has to exist.

### Lifecycle hooks

- `initialize!(model::AbstractModel)`: called exactly once per `run!` before the
  first time step. Use it to allocate buffers, fill halos, or pre-compute
  coefficients. The default implementation is a no-op.

- `update_state!(model, callbacks=[]; compute_tendencies=true)`: invoked by
  Simulation right after `initialize!` and inside most time steppers. This is
  where models fill halos, update boundary conditions, recompute auxiliary
  fields, and run callbacks with an `UpdateStateCallsite`. Implementations
  typically finish by calling `compute_tendencies!(model, callbacks)` so that
  any `TendencyCallsite` callbacks can modify tendencies before integration.

- `time_step!(model, Δt; callbacks=())`: advances the model clock and its
  prognostic variables by one step. Simulation hands in the tuple of
  `ModelCallsite` callbacks so the model can execute `TendencyCallsite` (before
  tendencies are applied) and `UpdateStateCallsite` callbacks (after auxiliary
  updates). The method must call `tick!(model.clock, Δt)` (or equivalent) so
  that `time(model)` and `iteration(model)` remain consistent.

- `set!(model, checkpoint_path_or_data)`: only required if the model supports
  being restarted via `run!(sim; pickup=...)`. Simulation forwards the `pickup`
  argument directly to `set!(sim.model, ...)`.

### Optional integrations

While not required for Simulation itself, the following methods enable the rest
of the Oceananigans ecosystem to “see” the model:

- `fields(model)` and `prognostic_fields(model)`: return `NamedTuple`s of
  fields so diagnostics, output writers, and NaN checkers know what to touch.

- `default_nan_checker(model)`: customize the `NaNChecker` that Simulation adds
  by default.

## Example: a zero-dimensional `LorenzModel`

The example below shows a deliberately tiny model that integrates the Lorenz
system on a 0D grid. The implementation demonstrates how little is required:
store a `grid` and `clock`, provide `time_step!` and `update_state!`
implementations, and rely on fallbacks for the rest.

### Implementing the interface

```@example model_interface
using Oceananigans
using Oceananigans.Simulations: Simulation, run!
using Oceananigans.TimeSteppers: Clock, tick!, update_state!, time_step!
using Oceananigans: TendencyCallsite, UpdateStateCallsite

mutable struct LorenzModel{FT, G, CLK} <: AbstractModel
    grid :: G
    clock :: CLK
    timestepper :: Nothing
    σ :: FT
    ρ :: FT
    β :: FT
    x :: FT
    y :: FT
    z :: FT
end

function LorenzModel(; σ = 10.0, ρ = 28.0, β = 8/3,
                     u0 = (1.0, 0.0, 0.0), FT = Float64)

    grid = RectilinearGrid(size = (1, 1, 1), extent = (1, 1, 1))
    clock = Clock{FT}(time = zero(FT))

    return LorenzModel{FT, typeof(grid), typeof(clock)}(
        grid, clock, nothing, FT(σ), FT(ρ), FT(β),
        FT(u0[1]), FT(u0[2]), FT(u0[3])
    )
end

Base.summary(::LorenzModel) = "LorenzModel"

@inline function lorenz_rhs(σ, ρ, β, x, y, z)
    return (σ * (y - x),
            x * (ρ - z) - y,
            x * y - β * z)
end

function rk4_step!(model::LorenzModel, Δt)
    σ, ρ, β = model.σ, model.ρ, model.β
    x₁, y₁, z₁ = model.x, model.y, model.z

    k₁x, k₁y, k₁z = lorenz_rhs(σ, ρ, β, x₁, y₁, z₁)
    k₂x, k₂y, k₂z = lorenz_rhs(σ, ρ, β,
                               x₁ + 0.5Δt * k₁x,
                               y₁ + 0.5Δt * k₁y,
                               z₁ + 0.5Δt * k₁z)
    k₃x, k₃y, k₃z = lorenz_rhs(σ, ρ, β,
                               x₁ + 0.5Δt * k₂x,
                               y₁ + 0.5Δt * k₂y,
                               z₁ + 0.5Δt * k₂z)
    k₄x, k₄y, k₄z = lorenz_rhs(σ, ρ, β,
                               x₁ + Δt * k₃x,
                               y₁ + Δt * k₃y,
                               z₁ + Δt * k₃z)

    model.x = x₁ + Δt / 6 * (k₁x + 2k₂x + 2k₃x + k₄x)
    model.y = y₁ + Δt / 6 * (k₁y + 2k₂y + 2k₃y + k₄y)
    model.z = z₁ + Δt / 6 * (k₁z + 2k₂z + 2k₃z + k₄z)

    return nothing
end

function update_state!(model::LorenzModel, callbacks = (); compute_tendencies = false)
    for callback in callbacks
        callback.callsite isa UpdateStateCallsite && callback(model)
    end
    return nothing
end

function time_step!(model::LorenzModel, Δt; callbacks = ())
    model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies = false)

    for callback in callbacks
        callback.callsite isa TendencyCallsite && callback(model)
    end

    rk4_step!(model, Δt)
    tick!(model.clock, Δt)

    update_state!(model, callbacks; compute_tendencies = false)
    return nothing
end
```

### Using the model inside Simulation

```@example model_interface
lorenz = LorenzModel()
sim = Simulation(lorenz; Δt = 0.01, stop_time = 0.5, verbose = false)
run!(sim)

(sim.model.x, sim.model.y, sim.model.z)
```

This minimal implementation inherits all other behavior from the generic
`AbstractModel` fallbacks: Simulation can query `time(sim.model)`, diagnostics
can read `sim.model.clock`, and callbacks scheduled on `ModelCallsite`s execute
because `time_step!` forwards the tuple that Simulation hands to it. Larger
models can follow the same recipe while swapping in sophisticated grids,
closures, and time steppers.

