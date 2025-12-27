# [Checkpointing](@id checkpointing)

A [`Checkpointer`](@ref) can be used to serialize the simulation state to a file from which
the simulation can be restored at any time. This is useful if you'd like to periodically checkpoint when running
long simulations in case of crashes or hitting cluster time limits, but also if you'd like to restore
from a checkpoint and try out multiple scenarios.

For example, to periodically checkpoint to disk every 1,000,000 seconds of simulation
time to files of the form `checkpoint_iteration12500.jld2` where `12500` is the iteration
number (automatically filled in).

Here's an example where we checkpoint every 5 iterations. This is far more often than appropriate for
typical applications: we only do it here for illustration purposes.

```@repl checkpointing
using Oceananigans

model = NonhydrostaticModel(grid=RectilinearGrid(size=(8, 8, 8), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=1, stop_iteration=8)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(5), prefix="model_checkpoint")
```

For long simulations, use `cleanup=true` to automatically delete old checkpoint files
when a new one is written, keeping only the latest checkpoint:

```julia
Checkpointer(model, schedule=IterationInterval(1000), prefix="checkpoint", cleanup=true)
```

Again, for illustration purposes of this example, we also add another callback so we can see the iteration
of the simulation

```@repl checkpointing
show_iteration(sim) = @info "iteration: $(iteration(sim)); time: $(prettytime(sim.model.clock.time))"
add_callback!(simulation, show_iteration, name=:info, IterationInterval(1))
```

Now let's run

```@repl checkpointing
run!(simulation)
```

The default options should provide checkpoint files that are easy to restore from (in most cases).
For more advanced options and features, see [`Checkpointer`](@ref).

## Picking up a simulation from a checkpoint file

Picking up a simulation from a checkpoint requires the original script that was used to generate
the checkpoint data. Change the first instance of [`run!`](@ref) in the script to take `pickup=true`.

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
set!(simulation, filepath)   # restore from specific file (no Checkpointer required)
set!(simulation, true)       # restore from latest checkpoint (requires Checkpointer)
set!(simulation, iteration)  # restore from specific iteration (requires Checkpointer)
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
