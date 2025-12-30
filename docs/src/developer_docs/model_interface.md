# Model interface

Oceananigans models are concrete subtypes of `AbstractModel` that can be advanced
by [`Simulation`](@ref Simulation). This page documents the *model interface*:
the handful of functions and data that a model must provide so that Simulation
can initialize it, march it forward, and coordinate callbacks, diagnostics, and
output writers.

The interface lives primarily inside `src/Simulations/` and is intentionally
minimal so that new models (or pedagogical toy models) can be written without
depending on the full ocean model infrastructure.

## The `AbstractModel` type

`AbstractModel{TS, A}` is parameterized by two type parameters:

- `TS`: The time-stepper type. This enables dispatch on models with specific
  time-stepping schemes. For example, `AbstractModel{<:QuasiAdamsBashforth2TimeStepper}`
  matches any model using the quasi-Adams-Bashforth second-order scheme.
  Models without a conventional time-stepper (e.g., the `LorenzModel` example below)
  can use `Nothing` for this parameter.

- `A`: The architecture type (e.g., `CPU`, `GPU`). This allows dispatch based
  on the computational backend.

Models that don't fit the conventional time-stepper pattern can use
`AbstractModel{Nothing, Nothing}` as their supertype.

## Lifecycle overview

When `run!(sim::Simulation)` is called the following high-level sequence occurs:

1. `initialize!(sim.model)` prepares the model state,
   etc.) and `update_state!(sim.model)` computes any auxiliary tendencies.

2. Time-stepping begins. For every time step, Simulation computes an aligned `Δt`, gathers callbacks
   that should run inside the model (`ModelCallsite`s), and calls
   `time_step!(sim.model, Δt; callbacks=model_callbacks)`. Most models will also call `update_state!`
   at the end of `time_step!`. This ensures that the auxiliary state (including halo regions, closure auxiliary fields, etc) is current with the prognostic state
   so that output and callbacks can execute correctly on a fully consistent model state.

3. After `time_step!(model, Δt)`, `Simulation` executes its diagnostics, output
   writers, and callbacks scheduled on the `TimeStepCallsite`.

Because `Simulation` assumes this protocol, any custom `AbstractModel` should
implement (or inherit sane fallbacks for) the items listed below.

## Structure and extensions of an `AbstractModel`

### Required properties

- `model.grid`: Simulation uses the grid to determine `architecture(model)` 
  and `eltype(model)` via the fallbacks `architecture(model) = model.grid.architecture`
  and `eltype(model) = eltype(model.grid)`.

- `model.clock :: Clock`: the source of truth for `time(model)` and
  `iteration(model)`. Simulation uses it for stop criteria and logging, and
  resets it via `reset_clock!(model)` when `reset!(sim)` is called.

### Lifecycle hooks

- `update_state!(model, callbacks=[]; compute_tendencies=true)`: invoked by
  Simulation right after `initialize!` and inside most time steppers. This is
  where models fill halos, update boundary conditions, recompute auxiliary
  fields, and run callbacks with an `UpdateStateCallsite`. Implementations
  typically finish by calling `compute_tendencies!(model, callbacks)` so that
  any `TendencyCallsite` callbacks can modify tendencies before integration.

- `time_step!(model, Δt; callbacks=[])`: advances the model clock and its
  prognostic variables by one step. Simulation hands in the tuple of
  `ModelCallsite` callbacks so the model can execute `TendencyCallsite` (before
  tendencies are applied) and `UpdateStateCallsite` callbacks (after auxiliary
  updates). The method must call `tick!(model.clock, Δt)` (or equivalent) so
  that `time(model)` and `iteration(model)` remain consistent.

- `set!(model, kw...)`: not strictly required, but strongly recommended as an
  interface for users to modify the model's prognostic state.

- `initialize!(model::AbstractModel)`: called exactly once per `run!` before the
  first time step.

### Optional integrations

While not required for Simulation itself, the following methods enable the rest
of the Oceananigans ecosystem to "see" the model:

- `timestepper(model)`: return the model's time-stepper object (or `nothing` if
  the model does not use a time-stepper). Simulation uses this to reset the
  time-stepper state when `reset!(sim)` is called. The default fallback returns
  `nothing`, so models without a time-stepper need not implement this method.
  Models with a time-stepper should implement this as
  `timestepper(model::MyModel) = model.timestepper`.

- `fields(model)` and `prognostic_fields(model)`: return `NamedTuple`s of
  fields so diagnostics, output writers, and NaN checkers know what to touch.

- `default_nan_checker(model)`: customize the `NaNChecker` that Simulation adds
  by default.

## Models that implement this interface

Several models across the CliMA ecosystem implement this interface:

**Oceananigans.jl**
- `NonhydrostaticModel`: solves the incompressible Navier-Stokes equations
- `HydrostaticFreeSurfaceModel`: solves the hydrostatic Boussinesq equations with a free surface
- `ShallowWaterModel`: solves the shallow water equations

**ClimaOcean.jl**
- `OceanSeaIceModel`: couples ocean, sea ice, and atmosphere components for Earth system modeling.
                      The components themselves may be `Simulation` that contain `AbstractModel`,
                      generating a nested structure.

**ClimaSeaIce.jl**
- `SeaIceModel`: simulates sea ice thermodynamics and dynamics

**Breeze.jl**
- `AtmosphereModel`: simulates atmospheric dynamics

## Example: a zero-dimensional `LorenzModel`

The example below shows a deliberately tiny model that integrates the Lorenz
system on a 0D grid. The implementation demonstrates how little is required:
store a `grid` and `clock`, provide `time_step!` and `update_state!`
implementations, and rely on fallbacks for the rest. Here we use an explicit
forward-Euler step so all of the logic lives directly inside `time_step!`.

### Implementing the interface

```@example model_interface
using Oceananigans
using Oceananigans.Models: AbstractModel
using Oceananigans.Simulations: Simulation, run!
using Oceananigans.TimeSteppers: Clock, tick!
import Oceananigans.TimeSteppers: update_state!, time_step!
import Oceananigans.Fields: set!
using Oceananigans: TendencyCallsite, UpdateStateCallsite

mutable struct LorenzModel{G, C, P, S} <: AbstractModel{Nothing, Nothing}
    grid :: G
    clock :: C
    parameters :: P
    state :: S
end

function LorenzModel(FT=Oceananigans.defaults.FloatType;
                     σ = 10, ρ = 28, β = 8/3)

    grid = RectilinearGrid(size=(), topology=(Flat, Flat, Flat))
    clock = Clock{FT}(time = zero(FT))
    parameters = (σ = FT(σ), ρ = FT(ρ), β = FT(β))
    state = (x = CenterField(grid), y = CenterField(grid), z = CenterField(grid))

    return LorenzModel(grid, clock, parameters, state)
end

Base.summary(::LorenzModel) = "LorenzModel"

function set!(model::LorenzModel; kw...)
    for (name, val) in kw
        set!(model.state[name], val)
    end
    return nothing
end

function update_state!(model::LorenzModel, callbacks = []; compute_tendencies=true)
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

    x, y, z = first(model.state.x), first(model.state.y), first(model.state.z)
    σ, ρ, β = model.parameters.σ, model.parameters.ρ, model.parameters.β

    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z

    @inbounds begin
        model.state.x[1, 1, 1] = x + Δt * dx
        model.state.y[1, 1, 1] = y + Δt * dy
        model.state.z[1, 1, 1] = z + Δt * dz
    end

    tick!(model.clock, Δt)
    update_state!(model, callbacks)

    return nothing
end
```

### Using the model inside Simulation

We set up a `Callback` to record the trajectory at each time step:

```@example model_interface
lorenz = LorenzModel()
set!(lorenz, x=1)
simulation = Simulation(lorenz; Δt=0.01, stop_time=100, verbose=false)

trajectory = NTuple{3, Float64}[]

function record_trajectory!(sim)
    x = first(sim.model.state.x)
    y = first(sim.model.state.y)
    z = first(sim.model.state.z)
    push!(trajectory, (x, y, z))
end

add_callback!(simulation, record_trajectory!)
run!(simulation)
nothing # hide
```

Finally, we visualize the famous Lorenz attractor with a 3D line plot:

```@example model_interface
using CairoMakie

fig = Figure(size=(600, 500))

ax = Axis3(fig[1, 1];
           xlabel="x", ylabel="y", zlabel="z",
           title="Lorenz attractor",
           azimuth=1.2π)

xs = [p[1] for p in trajectory]
ys = [p[2] for p in trajectory]
zs = [p[3] for p in trajectory]

lines!(ax, xs, ys, zs; linewidth=0.5, color=zs, colormap=:magma)

fig
```

This minimal implementation inherits all other behavior from the generic
`AbstractModel` fallbacks: Simulation can query `time(sim.model)`, diagnostics
can read `sim.model.clock`, and callbacks scheduled on `ModelCallsite`s execute
because `time_step!` forwards the tuple that Simulation hands to it. Larger
models can follow the same recipe while swapping in sophisticated grids,
closures, and time steppers.

