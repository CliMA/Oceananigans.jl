# [Callbacks](@id callbacks)

A [`Callback`](@ref) can be used to execute an arbitrary user-defined function on the
simulation at user-defined times.

For example, we can specify a callback which displays the run time every 2 iterations:
```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```@example callbacks
using Oceananigans

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = NonhydrostaticModel(; grid)
simulation = Simulation(model, Δt=1, stop_iteration=10)

show_time(sim) = @info "Time is $(prettytime(sim.model.clock.time))"
simulation.callbacks[:pretty_basic] = Callback(show_time, IterationInterval(2))

simulation
```

Now when simulation runs the simulation the callback is called.

```@example callbacks
run!(simulation)
```

We can also use the convenience [`add_callback!`](@ref):

```@example callbacks
add_callback!(simulation, show_time, IterationInterval(2))

simulation
```

The keyword argument `callsite` determines the moment at which the callback is executed.
By default, `callsite = TimeStepCallsite()`, indicating execution _after_ the completion of
a timestep. This is the only callsite that is owned by `Simulation`.

Other callsite options are `callsite = TendencyCallsite()` and `callsite = UpdateStateCallsite()`.
When these callsites are used, the callback is a function of the `model` in question rather than `Simulation`.
These callsites must be implemented by models: for `NonhydrostaticModel` and `HydrostaticFreeSurfaceModel`,
`TendencyCallsite` callbacks are executed after computing tendencies but _before_ taking a timestep.
`UpdateStateCallsite` callbacks are executed within `update_state!`, after auxiliary variables have been computed
(for multi-stage time-steppers, `update_state!` may be called multiple times per timestep).

As an example of a callback with `callsite = TendencyCallsite()` , we show below how we can
manually add to the tendency field of one of the velocity components. Here we've chosen
the `:u` field using parameters:

```@example callbacks
using Oceananigans
using Oceananigans: TendencyCallsite

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = NonhydrostaticModel(; grid)
simulation = Simulation(model, Δt=1, stop_iteration=10)

function modify_tendency!(model, params)
    Gⁿc = model.timestepper.Gⁿ[params.c]
    parent(Gⁿc) .+= params.δ
    return nothing
end

add_callback!(simulation,
              modify_tendency!, 
              callsite = TendencyCallsite(),
              parameters = (c = :u, δ = 1))
                                           

run!(simulation)
```

Above there is no forcing at all, but due to the callback the ``u``-velocity is increased.

```@example callbacks
@info model.velocities.u
```

!!! note "Example only for illustration purposes"
    The above is a redundant example since it could be implemented better with a simple forcing function.
    We include it here though for illustration purposes of how one can use callbacks.

## Functions

Callback functions can only take one or two parameters `sim` - a simulation, or `model` for state callbacks, and optionally may also accept a NamedTuple of parameters.

## Scheduling

Callbacks rely on the schedules described in [Scheduling callbacks and output writers](@ref callback_schedules).
The most common choices are:
 - [`IterationInterval`](@ref): runs every `n` iterations
 - [`TimeInterval`](@ref): runs every `n` seconds of model time
 - [`SpecifiedTimes`](@ref): runs at the specified times
 - [`WallTimeInterval`](@ref): runs every `n` seconds of wall time
