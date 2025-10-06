# Callbacks

Callbacks let you weave custom logic into the simulation loop. They are plain Julia functions that
`Simulation` executes on a schedule you control. Typical uses include logging, adaptive control of
parameters (for example the time step), triggering outputs, or stopping a run once a condition is met.

```@meta
DocTestSetup = quote
    using Oceananigans
    using Oceananigans.Utils: prettytime, minute, ConsecutiveIterations
end
```

## Basic usage

Attach callbacks directly to `simulation.callbacks` or use [`add_callback!`](@ref) to register them
with a schedule. The example below logs progress every two iterations.

```@example callback_basics
grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = NonhydrostaticModel(; grid)

simulation = Simulation(model; Δt=1, stop_iteration=6)

show_time(sim) = @info "t = $(prettytime(sim.model.clock.time))"

add_callback!(simulation, show_time, IterationInterval(2); name = :progress)

run!(simulation)
```

Callbacks appear in the ordered dictionary `simulation.callbacks`. Each entry stores the function,
its schedule, callsite, and optional parameters:

```@example callback_basics
simulation.callbacks[:progress]
```

## Call sites and parameterized callbacks

Callbacks run at the time-step callsite by default, meaning immediately after a step completes.
Other call sites expose different parts of the integrator:

- [`TendencyCallsite`](@ref Oceananigans.TendencyCallsite): after tendencies are assembled but before the state advances.
- [`UpdateStateCallsite`](@ref Oceananigans.UpdateStateCallsite): inside `update_state!`, which can execute multiple times per step for multi-stage integrators.

Callbacks optionally accept a NamedTuple of parameters. This is useful when the same function should
operate on different fields or thresholds.

```@example callback_basics
using Oceananigans: TendencyCallsite

grid2 = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
model2 = NonhydrostaticModel(; grid = grid2)
simulation2 = Simulation(model2; Δt = 1, stop_iteration = 4)

function boost_tendency!(sim, params)
    sim.model.timestepper.Gⁿ[params.component] .+= params.increment
    return nothing
end

simulation2.callbacks[:nudging] = Callback(boost_tendency!, IterationInterval(1);
                                           callsite = TendencyCallsite(),
                                           parameters = (component = :u, increment = 1.0))

run!(simulation2)
```

!!! note "Callbacks vs. forcing"
    The example above manipulates the tendency for illustration only. In production work a forcing term or
    closure is often a clearer option; callbacks shine when you need logic that depends on the state of the
    whole simulation or that operates outside of the model equations.

## Scheduling callback execution (@id callback_schedules)

Schedules control _when_ a callback fires. Oceananigans provides several stock schedules that cover
iteration counts, model time, wall-clock time, and explicit lists of timestamps:

- [`IterationInterval(n)`](@ref): actuates every `n` iterations.
- [`TimeInterval(Δt)`](@ref): actuates every `Δt` seconds of model time.
- [`SpecifiedTimes(times...)`](@ref): actuates whenever the model clock reaches one of the supplied times.
- [`WallTimeInterval(Δt)`](@ref): actuates based on wall-clock seconds.
- [`AveragedTimeInterval`](@ref AveragedTimeInterval): actuates periodically while also accumulating a moving time average (used mainly by output writers).

Schedules can be combined or embellished:

- [`ConsecutiveIterations(parent, N)`](@ref Oceananigans.Utils.ConsecutiveIterations) fires on the parent schedule and the following `N` iterations.
- [`AndSchedule`](@ref Oceananigans.Utils.AndSchedule) requires **all** child schedules to actuate at once.
- [`OrSchedule`](@ref Oceananigans.Utils.OrSchedule) actuates when **any** child schedule does.

```@example callback_schedules
grid3 = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
model3 = NonhydrostaticModel(; grid = grid3)
simulation = Simulation(model3; Δt = 1, stop_iteration = 10)

log_times = SpecifiedTimes(3, 7)
burst_schedule = ConsecutiveIterations(log_times, 1) # fire at 3, 4, 7, 8
combined = OrSchedule(burst_schedule, TimeInterval(5))

add_callback!(simulation,
              sim -> @info "iteration $(sim.model.clock.iteration) at t=$(sim.model.clock.time)",
              combined;
              name = :timeline)

run!(simulation)
```

!!! tip "Time-step alignment"
    `Simulation` aligns time steps to meet scheduled events when `align_time_step = true` (the default).
    This ensures that `TimeInterval` and `SpecifiedTimes` callbacks execute at the requested instant. Set
    `align_time_step = false` to keep a fixed `Δt` even if that skips past a scheduled time.

When reusing a schedule object across multiple callbacks or simulations, call [`materialize_schedule`](@ref Oceananigans.Utils.materialize_schedule)
first. Stateful schedules such as `TimeInterval` track their last firing time; copying them resets the state.

## Helper constructors

Create callbacks manually with the [`Callback`](@ref) constructor when you need to specify call sites or parameters.
For simple functions, `add_callback!` builds the `Callback` for you. Both approaches store the result in
`simulation.callbacks`, so you can inspect or modify it later. `Callback` objects also support `initialize!(callback, simulation)`
and `finalize!(callback, simulation)` hooks that run at the beginning and end of `run!` when you specialize them for your
callback type.

Callbacks tie together the various parts of a simulation: they can monitor diagnostics, adjust time stepping
with a [`TimeStepWizard`](@ref), or coordinate with [output writers](@ref output_writers) to stamp metadata into each file.
