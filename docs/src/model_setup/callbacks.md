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
a timestep. The other options are `callsite = TendencyCallsite()` that executes the callback
after the tendencies are computed but _before_ taking a timestep and `callsite = UpdateStateCallsite()`
that executes the callback within `update_state!`, after auxiliary variables have been computed
(for multi-stage time-steppers, `update_state!` may be called multiple times per timestep).

As an example of a callback with `callsite = TendencyCallsite()` , we show below how we can
manually add to the tendency field of one of the velocity components. Here we've chosen
the `:u` field using parameters:

```@example checkpointing
using Oceananigans

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=1, stop_iteration=10)

function modify_tendency!(model, params)
    model.timestepper.Gⁿ[params.c] .+= params.δ
    return nothing
end

simulation.callbacks[:modify_u] = Callback(modify_tendency!, IterationInterval(1),
                                           callsite = TendencyCallsite(),
                                           parameters = (c = :u, δ = 1))

run!(simulation)
```

Above there is no forcing at all, but due to the callback the ``u``-velocity is increased.

```@example checkpointing
@info model.velocities.u
```

!!! note "Example only for illustration purposes"
    The above is a redundant example since it could be implemented better with a simple forcing function.
    We include it here though for illustration purposes of how one can use callbacks.

## Functions

Callback functions can only take one or two parameters `sim` - a simulation, or `model` for state callbacks, and optionally may also accept a NamedTuple of parameters.

## Scheduling

The time that callbacks are called at are specified by schedule functions which can be:
 - [`IterationInterval`](@ref) : runs every `n` iterations
 - [`TimeInterval`](@ref) : runs every `n`s of model run time
 - [`SpecifiedTimes`](@ref) : runs at the specified times
 - [`WallTimeInterval`](@ref) : runs every `n`s of wall time
