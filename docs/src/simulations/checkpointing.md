# [Checkpointing](@id checkpointing)

A [`Checkpointer`](@ref) can be used to serialize the entire model state to a file from which the model
can be restored at any time. This is useful if you'd like to periodically checkpoint when running
long simulations in case of crashes or hitting cluster time limits, but also if you'd like to restore
from a checkpoint and try out multiple scenarios.

For example, to periodically checkpoint the model state to disk every 1,000,000 seconds of simulation
time to files of the form `model_checkpoint_iteration12500.jld2` where `12500` is the iteration
number (automatically filled in).

Here's an example where we checkpoint every 5 iterations. This is far more often than appropriate for
typical applications: we only do it here for illustration purposes.

```@repl checkpointing
using Oceananigans

model = NonhydrostaticModel(grid=RectilinearGrid(size=(8, 8, 8), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=1, stop_iteration=8)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(5), prefix="model_checkpoint")
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

When `pickup=true` is provided to `run!` then it finds the latest checkpoint file in the current working
directory, loads prognostic fields and their tendencies from file, resets the model clock and iteration,
to the clock time and iteration that the checkpoint corresponds to, and updates the model auxiliary state.
After that, the time-stepping loop. In this simple example, although the simulation run up to iteration 8,
the latest checkpoint is associated with iteration 5.

```@repl checkpointing
simulation.stop_iteration = 12

run!(simulation, pickup=true)
```

Use `pickup=iteration`, where `iteration` is an `Integer`, to pick up from a specific iteration.
Or, use `pickup=filepath`, where `filepath` is a string, to pickup from a specific file located
at `filepath`.
