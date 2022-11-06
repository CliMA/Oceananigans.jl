# Callbacks

Callbacks can be used to execute an arbitrary user defined function on the simulation at user 
defined times, or between every time stepper sub-step for `state_callbacks`.

For example, we can specify a callback which displays the run time every 2 iterations:
```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```@repl checkpointing
using Oceananigans

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=1, stop_iteration=10)

show_time(sim) = @info "Time is $(prettytime(sim.model.clock.time))"

simulation.callbacks[:total_A] = Callback(show_time, IterationInterval(2))

run!(simulation)
```

State callbacks are useful for inter step modification of the model state (for example if you wanted to manually modify the tendency fields). Irrespective of the specified scheduling state callbacks are executed at every sub-step. As an example we can manually add to the tendency field of one of the velocity components, here I've chosen the `:u` field using parameters:

```@repl checkpointing
using Oceananigans

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))

function modify_tracer(model, params)
    model.timestepper.Gⁿ[params.c] .+= params.δ
end

model.state_callbacks[:modify_u] = Callback(modify_tracer, IterationInterval(1), (c = :u, δ = 1))

simulation = Simulation(model, Δt=1, stop_iteration=10)

run!(simulation)

@info model.velocities.u
```
Here there is no forcing, but due to the callback the velocity is increased. 
>This is a redundant example and here for illustration only, it could be implemented better with a simple forcing function.

## Functions

Callback functions can only take one or two parameters `sim` - a simulation, or `model` for state callbacks, and optionally may also accept a NamedTuple of parameters.

## Scheduling

The time that callbacks are called at are specified by schedule functions which can be:
 - [`IterationInterval`](@ref) : runs every `n` iterations
 - [`TimeInterval`](@ref) : runs every `n`s of model run time
 - [`SpecifiedTimes`](@ref) : runs at the specified times
 - [`WallTimeInterval`](@ref) : runs every `n`s of wall time


```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```@repl checkpointing
using Oceananigans, Oceananigans.Units

model = NonhydrostaticModel(grid=RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=1, stop_iteration=1)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(5years), prefix="model_checkpoint")

run!(simulation)
```

The default options should provide checkpoint files that are easy to restore from in most cases.
For more advanced options and features, see [`Checkpointer`](@ref).

## Picking up a simulation from a checkpoint file

Picking up a simulation from a checkpoint requires the original script that was used to generate
the checkpoint data. Change the first instance of [`run!`](@ref) in the script to take `pickup=true`:

```@repl checkpointing
simulation.stop_iteration = 2

run!(simulation, pickup=true)
```

which finds the latest checkpoint file in the current working directory (in this trivial case,
this is the checkpoint associated with iteration 0), loads prognostic fields and their tendencies
from file, resets the model clock and iteration, and updates the model auxiliary state before
starting the time-stepping loop.

Use `pickup=iteration`, where `iteration` is an `Integer`, to pick up from a specific iteration.
Or, use `pickup=filepath`, where `filepath` is a string, to pickup from a specific file located
at `filepath`.
