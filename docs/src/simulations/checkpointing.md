# [Checkpointing](@id checkpointing)

A [`Checkpointer`](@ref) can be used to serialize the simulation state to a file from which
the simulation can be restored at any time. This is useful if you'd like to periodically checkpoint when running
long simulations in case of crashes or hitting cluster time limits, but also if you'd like to restore
from a checkpoint and try out multiple scenarios.

Here's an example where we checkpoint every 5 iterations to files of the form
`model_checkpoint_iteration5.jld2` (where the iteration number is automatically included in
the filename). This is far more often than appropriate for typical applications: we only do
it here for illustration purposes.

```@repl checkpointing
using Oceananigans

model = NonhydrostaticModel(RectilinearGrid(size=(8, 8, 8), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=1, stop_iteration=8)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(5), prefix="model_checkpoint")
```

Use `cleanup=true` to automatically delete old checkpoint files
when a new one is written, keeping only the latest checkpoint:

```julia
Checkpointer(model, schedule=IterationInterval(1000), prefix="checkpoint", cleanup=true)
```

Again, for illustration purposes, we also add a callback so we can see the simulation progress:

```@repl checkpointing
show_iteration(sim) = @info "iteration: $(iteration(sim)), time: $(prettytime(sim.model.clock.time))"
add_callback!(simulation, show_iteration, name=:info, schedule=IterationInterval(1))
```

Now let's run

```@repl checkpointing
run!(simulation)
```

The default options should provide checkpoint files that are easy to restore from (in most cases).
For more advanced options and features, see [`Checkpointer`](@ref).

Checkpointing is supported for: `ShallowWaterModel`, `NonhydrostaticModel`, and `HydrostaticFreeSurfaceModel`
(including split-explicit, implicit, and explicit free surfaces, as well as z-star vertical coordinates).

## Picking up a simulation from a checkpoint file

Picking up a simulation from a checkpoint requires recreating the simulation identically
to how it was originally configured. This means using the same grid, model type, boundary
conditions, forcing, closures, and output writers. Only the **prognostic state** (data that
evolves during simulation) is restored from the checkpoint - not the simulation configuration.

When `pickup=true` is provided to `run!`, it finds the latest checkpoint file in the `Checkpointer`'s
directory, restores the simulation state (including model fields, clock, and timestepper state),
and then continues the time-stepping loop. In this simple example, although the simulation ran up to
iteration 8, the latest checkpoint is associated with iteration 5.

```@repl checkpointing
simulation.stop_iteration = 12

run!(simulation, pickup=true)
```

Use `pickup=iteration`, where `iteration` is an `Integer`, to pick up from a specific iteration.
Or, use `pickup=filepath`, where `filepath` is a string, to pickup from a specific file located
at `filepath`.

The [`set!`](@ref) function can also be used to restore from a checkpoint without immediately
running the simulation:

```julia
set!(simulation; checkpoint="path/to/file.jld2")  # restore from specific file
set!(simulation; checkpoint=:latest)               # restore from latest checkpoint (requires Checkpointer)
set!(simulation; iteration=12345)                  # restore from specific iteration (requires Checkpointer)
```

## Checkpointing on wall-clock time

For cluster jobs with time limits, use `WallTimeInterval` to checkpoint based on elapsed
wall-clock time rather than simulation time or iterations:

```julia
# Checkpoint every 30 minutes of wall-clock time
Checkpointer(model, schedule=WallTimeInterval(30minute), prefix="checkpoint")
```

This ensures checkpoints are saved regularly even if individual time steps vary significantly.

## Manual checkpointing

Use [`checkpoint`](@ref) to manually save the simulation state at any point:

```julia
checkpoint(simulation)                            # uses Checkpointer settings if available
checkpoint(simulation, filepath="my_state.jld2")  # write to specific file
```

If a `Checkpointer` is configured in `simulation.output_writers`, it will be used (respecting
its `dir`, `prefix`, and other settings). Otherwise, the checkpoint is written to the specified
`filepath`, or to `checkpoint_iteration{N}.jld2` in the current directory.

## Automatic checkpointing at end

Use `checkpoint_at_end=true` to automatically checkpoint the simulation when it finishes:

```julia
run!(simulation, checkpoint_at_end=true)  # Checkpoints when done
```

This ensures the final simulation state is saved, even if the simulation stops due to
wall time limits or other callbacks.

If a `Checkpointer` is configured, it will be used. Otherwise, a file named
`checkpoint_iteration{N}.jld2` is created in the current directory.

## What gets checkpointed

Checkpointing saves the **prognostic state** which is data that evolves during simulation. This includes
prognostic model fields (velocities, tracers, diffusivities, etc.), the clock, the state of the
time stepper, output writer state, turbulence closure state, free surface state, and Lagrangian particle
properties.

Static configuration is **not** checkpointed. This includes the grid, boundary conditions, forcing
functions, closure parameters, model options, and callbacks.

This means your script must recreate the simulation with identical configuration before
restoring from a checkpoint.
