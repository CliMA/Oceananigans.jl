# Callbacks

A [`Callback`](@ref) can be used to execute an arbitrary user-defined function on the
simulation at user-defined times.

For example, we can specify a callback which displays the run time every 2 iterations:
```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```@example checkpointing
using Oceananigans

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=1, stop_iteration=10)

show_time(sim) = @info "Time is $(prettytime(sim.model.clock.time))"

simulation.callbacks[:total_A] = Callback(show_time, IterationInterval(2))

simulation
```

Now when simulation runs the simulation the callback is called.

```@example checkpointing
run!(simulation)
```

We can also use the convenience [`add_callback!`](@ref):

```@example checkpointing
add_callback!(simulation, show_time, name=:total_A_via_convenience, IterationInterval(2))

simulation
```

The keyword argument `callsite` determines the moment at which the callback is executed.
By default, `callsite = TimeStepCallsite()`, indicating execution _after_ the completion of
a timestep. The other options is `callsite = TendencyCallsite()` that executes the callback
after the tendencies are computed but _before_ taking a timestep.

As an example of the latter, we use a callback to manually add to the tendency field of one
of the velocity components, here we've chosen the `:u` field using parameters:

```@example checkpointing
using Oceananigans

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=1, stop_iteration=10)

function modify_tracer!(model, params)
    model.timestepper.Gⁿ[params.c] .+= params.δ
    return nothing
end

simulation.callbacks[:modify_u] = Callback(modify_tracer!, IterationInterval(1),
                                           callsite = TendencyCallsite(),
                                           parameters = (c = :u, δ = 1))

run!(simulation)

@info model.velocities.u
```

Above there is no forcing at all, but due to the callback the velocity is increased.

> This is a redundant example and here for illustration only, it could be implemented
  better with a simple forcing function.

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

```@example checkpointing
using Oceananigans, Oceananigans.Units

model = NonhydrostaticModel(grid=RectilinearGrid(size=(8, 8, 8), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=1, stop_iteration=1)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(5days), prefix="model_checkpoint")

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
this is the checkpoint associated with iteration 0), loads the prognostic fields and their tendencies
from file, resets the model clock and iteration, and updates the model auxiliary state before
starting the time-stepping loop.

Use `pickup=iteration`, where `iteration` is an `Integer`, to pick up from a specific iteration.
Otherwise, use `pickup=filepath`, where `filepath` is a string, to pickup from a specific file located
at `filepath`.
