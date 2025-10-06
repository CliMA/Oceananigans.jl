# Simulations: orchestrating runs (@id simulation_overview)

`Simulation` objects wrap a model in the control logic that actually pushes it forward in time.
They manage the time-stepping loop, stop conditions, output, diagnostics, and callbacks so that
Oceananigans models can focus on physics while simulations focus on workflow.

Ocean modelling scripts usually follow this pattern:

1. Build a grid and a model
2. Wrap the model in a `Simulation`
3. Attach output writers, diagnostics, and callbacks
4. Call [`run!`](@ref) to integrate in time

The sections below sketch this lifecycle with a few small examples and point to the dedicated
pages on callbacks and output writers.

```@meta
DocTestSetup = quote
    using Oceananigans
    using Oceananigans.Diagnostics: AdvectiveCFL
    using Oceananigans.Utils: prettytime
end
```

## A minimal simulation

```@example simulation_overview
using Oceananigans

arch = CPU()
grid = RectilinearGrid(arch; size=(8, 8, 8), extent=(128, 128, 64))
model = NonhydrostaticModel(; grid, tracers=(:T,))

simulation = Simulation(model; Δt=10.0, stop_iteration=20)
run!(simulation)
simulation
```

`Simulation` stores bookkeeping such as total iterations performed, the next `Δt`, and the
ordered dictionaries `callbacks`, `output_writers`, and `diagnostics`. In the example above we
run for only 20 steps; realistic applications chain extra logic onto this base object.

## Stop criteria and time-step control

The `Simulation` constructor accepts several stopping conditions. The most common are:

- `stop_iteration`: maximum number of steps
- `stop_time`: maximum model clock time (same units as the model's clock)
- `wall_time_limit`: maximum wall-clock seconds before the run aborts

```@example simulation_overview
simulation = Simulation(model; Δt=15, stop_time=60, wall_time_limit=30)
simulation
```

Oceananigans can adapt the time-step by inserting a [`TimeStepWizard`](@ref) as a callback.
The wizard monitors advective and diffusive Courant numbers, shrinking or growing `Δt` while
respecting bounds such as `max_change` or `max_Δt`.

```@example simulation_overview
wizard = TimeStepWizard(cfl = 0.7, max_change = 1.1, max_Δt = 20.0)
add_callback!(simulation, wizard; name = :wizard)
simulation.callbacks[:wizard]
```

By default `Simulation` aligns the final substep of each `Δt` so that scheduled events (like
output) land exactly on their target times. Setting `align_time_step = false` allows the
simulation to march with a fixed `Δt` even if that means overshooting a scheduled time.

## Monitoring progress with callbacks

Callbacks are small pieces of code that run on a [schedule](@ref callback_schedules).
They automate tasks such as logging, runtime adjustments, or scientific diagnostics.

```@example simulation_overview
show_progress(sim) = @info "step $(sim.model.clock.iteration): t = $(prettytime(sim.model.clock.time))"

add_callback!(simulation, show_progress, IterationInterval(5); name = :progress)
run!(simulation)
```

The `callsite` keyword lets callbacks hook into different parts of the time-stepping cycle.
For example, use `callsite = TendencyCallsite()` to modify tendencies before a step, or
`UpdateStateCallsite()` to react after auxiliary variables update. See the [Callbacks page](@ref callbacks)
for more customization patterns, including parameterized callbacks and state callbacks.

## Writing output

Output writers live in the ordered dictionary `simulation.output_writers`. Each writer pairs
outputs with a schedule describing _when_ to export them and a backend (NetCDF, JLD2, or
Checkpointer) describing _how_ to serialize them. Time-step alignment ensures the writer's
schedule is honoured without user intervention.

```@example simulation_overview
using Oceananigans.OutputWriters
using NCDatasets

fields = Dict("u" => simulation.model.velocities.u)

simulation.output_writers[:surface_slice] = NetCDFWriter(simulation.model, fields;
                                                         filename = "demo.nc",
                                                         schedule = TimeInterval(30.0),
                                                         indices = (:, :, grid.Nz))
simulation.output_writers
```

During `run!`, Oceananigans calls each writer whenever its schedule actuates. Detailed usage
examples appear in the [Output writers page](@ref output_writers).

## Diagnostics

Diagnostics behave a lot like callbacks: place them in `simulation.diagnostics` and update them
on a schedule. A typical workflow creates a diagnostic, associates it with a callback, and
optionally writes the results via an output writer.

```@example simulation_overview
simulation = Simulation(model; Δt=10, stop_iteration=15)

cfl = AdvectiveCFL(simulation.Δt)
add_callback!(simulation, cfl; name = :cfl_monitor)

run!(simulation)
```

Diagnostics are evaluated but not stored automatically. If you want to accumulate values or
flush them to disk, wrap the diagnostic call in a callback that records the values or writes
through an output writer.

## Putting it together

Combining callbacks, output writers, and adaptive time-stepping turns a few lines of model code
into a robust workflow. The snippet below shows a compact pattern frequently used in production.

```@example simulation_overview
using Oceananigans.Utils: hour, minute

model = NonhydrostaticModel(; grid, tracers = (:T,))
simulation = Simulation(model; Δt = 20.0, stop_time = 2hour)

progress(sim) = @info "t = $(prettytime(sim.model.clock.time)), Δt = $(prettytime(sim.Δt))"
add_callback!(simulation, progress, TimeInterval(15minute); name = :progress)

wizard = TimeStepWizard(cfl = 0.8, max_change = 1.2, max_Δt = 60.0)
add_callback!(simulation, wizard; name = :wizard)

simulation.output_writers[:snapshots] = JLD2Writer(simulation.model, simulation.model.velocities;
                                                   filename = "snapshots.jld2",
                                                   schedule = TimeInterval(30minute))

run!(simulation)
```

This pattern combines short informative logging, adaptive stepping, and periodic state dumps.
Modify the schedules or outputs to suit your experiment. For more recipes continue with the
pages on [Callbacks](@ref callbacks) and [Output writers](@ref output_writers).
