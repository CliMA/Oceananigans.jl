# [Simulations: managing model time-stepping](@id simulation_overview)

[`Simulation`](@ref Oceananigans.Simulations.Simulation) orchestrates model time-stepping loops that include
stop conditions, writing output, and the execution of "callbacks" that can do everything
from monitoring simulation progress to editing the model state.

Scripts for numerical experiments can be broken into four parts:

1. Definition of the "grid" and physical domain for the numerical experiment
2. Configuration of physics and model construction
3. Building a `Simulation` time-stepping loop with callbacks and output writers
4. Calling [`run!`](@ref) to integrate the model forward in time.

`Simulation` is the final boss object that users interact
with in order in the process of performing a numerical experiment.

## A simple simulation

```@meta
DocTestSetup = quote
    using Oceananigans
    using Oceananigans.Diagnostics: AdvectiveCFL
    using Oceananigans.Utils: prettytime
end
```

A minimal example illustrates how `Simulation` works:

```@example simulation_overview
using Oceananigans

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid)
simulation = Simulation(model; Δt=7, stop_iteration=6)
run!(simulation)

simulation
```

A more complicated setup might invoke multiple callbacks:

```@example simulation_overview
using Oceananigans

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid)
simulation = Simulation(model; Δt=7, stop_time=14)

print_progress(sim) = @info string("Iteration: ", iteration(sim), ", time: ", time(sim))
add_callback!(simulation, print_progress, IterationInterval(2))

declare_time(sim) = @info string("The simulation has been running for ", prettytime(sim.run_wall_time), "!")
add_callback!(simulation, declare_time, TimeInterval(10))

run!(simulation)
```

`Simulation` book-keeps the total iterations performed, the next `Δt`, and the
lists of `callbacks` and `output_writers`.
`Simulation`s can be continued, which is helpful for interactive work:

```@example simulation_overview
simulation.stop_time = 42
run!(simulation)
```

## Stop criteria and time-step control

The `Simulation` constructor accepts three stopping conditions:

- `stop_iteration`: maximum number of steps
- `stop_time`: maximum model clock time (same units as the model's clock)
- `wall_time_limit`: maximum wall-clock seconds before the run aborts

```@example simulation_overview
simulation = Simulation(model; Δt=0.1, stop_time=10, stop_iteration=10000, wall_time_limit=30)
```

### Callback basics

[Callback](@ref) executes arbitrary code on a [schedule](@ref callback_schedules).
Callbacks can be used to monitor simulation progress, compute diagnostics, and adjust the
course of a simulation.
To illustrate a hierarchy of callbacks we use a simulation with a forced passive tracer:

```@example simulation_overview
using Oceananigans

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))

c_source(x, y, z, t, c) = 0.1 * c
c_forcing = Forcing(c_source, field_dependencies=:c)
model = NonhydrostaticModel(grid; tracers=:c, forcing=(; c=c_forcing))
simulation = Simulation(model; Δt=0.1, stop_time=10)

# Add a callback that prints progress
print_progress(sim) = @info "Iter $(iteration(sim)): $(prettytime(sim))"
add_callback!(simulation, print_progress, IterationInterval(25), name=:progress)
run!(simulation)
```

!!! note "Naming callbacks"
    Callbacks can optionally be assigned a `name` via `add_callback!` or by adding
    the callback manually to the `callbacks` dictionary: `simulation.callbacks[:mine] = my_callback`.
    Names can be used to identify, modify, or delete callbacks from `Simulation`.

### Callbacks: for stopping a simulation

To spark your imagination, consider that callbacks can be used to implement
arbitrary stopping criteria. As an example we consider a stopping criteria based on the magnitude of a tracer:

```@example simulation_overview
using Printf

set!(model, c=1)

function stop_simulation(sim)
    if maximum(sim.model.tracers.c) >= 2
        sim.running = false
        @info "The simulation is stopping because max(c) >= 2."
    end
    return nothing
end

# The default schedule is IterationInterval(1)
add_callback!(simulation, stop_simulation)

function print_progress(sim)
    max_c = maximum(sim.model.tracers.c)
    @info @sprintf("Iter %d: t = %s, max(c): %.2f",
                   iteration(sim), prettytime(sim), max_c)
    return nothing
end

add_callback!(simulation, stop_simulation, IterationInterval(10), name=:progress)
simulation.stop_time += 10
run!(simulation)
```

!!! note "Callback execution order"
    Callbacks are executed in the order they appear within the `callbacks` `OrderedDict`,
    which in turn corresponds to the order they are added.

### Adaptive time-stepping with `TimeStepWizard`

The time-step can be changed by modifying `simulation.Δt`.
To decrease the computational cost of simulations of flows that significantly grow or decay in time,
users may invoke a special callback called [`TimeStepWizard`](@ref) (which is associated with a special
helper function [`conjure_time_step_wizard!`](@ref)).
`TimeStepWizard` monitors the
[advective and diffusive Courant numbers](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition),
increasing or decreasing `simulation.Δt` to keep the time-step near its maximum stable value
while respecting bounds such as `max_change` or `max_Δt`.

```@example simulation_overview
ϵ(x, y, z) = 1e-2
set!(model, c=1, u=ϵ, v=ϵ, w=ϵ)

conjure_time_step_wizard!(simulation, cfl=0.7)
simulation
```

A `TimeStepWizard` has been added to the list of `simulation.callbacks`.
`TimeStepWizard` is authorized to modify the time-step as the simulation progresses
(every 10 iterations by default, but this can be modified):

```@example simulation_overview
print_progress(sim) = @info string("Iter: ", iteration(sim), ": Δt = ", prettytime(sim.Δt))
add_callback!(simulation, print_progress, IterationInterval(10), name=:progress)
simulation.stop_time += 1
run!(simulation)
```

### Time-step alignment with scheduled events

By default, `Simulation` "aligns" `Δt` so that events which are _scheduled by time_ (such as `TimeInterval` and `SpecifiedTimes`) occur exactly on schedule.
Time-step alignment may be disabled by setting `align_time_step = false` in the `Simulation` constructor.

```@example simulation_overview
print_progress(sim) = @info "At t = $(time(sim)), iter = $(iteration(sim))"
add_callback!(simulation, print_progress, TimeInterval(0.2), name=:progress)
simulation.stop_time += 0.6
run!(simulation)
```

!!! note "Minimum relative step"
    Due to round-off error, time-step alignment can produce very small `Δt` close to machine
    epsilon --- for example, when `TimeInterval` is a constant multiple of a fixed `Δt`.
    In some cases, `Δt` close to machine epsilon is undesirable: for example, with `NonhydrostaticModel`
    a machine epsilon `Δt` will produce a pressure field that is polluted by numerical error due to
    its pressure correction algorithm. To mitigate issues associated with very small time-steps,
    the `Simulation` constructor accepts a `minimum_relative_step` argument (a typical choice is 1e-9).
    When `minimum_relative_step > 0`, a time-step will be skipped (instead, the clock is simply reset)
    if an aligned  time-step is less than `minimum_relative_step * previous_Δt`.

### Callback "callsites"

The `callsite` keyword lets callbacks hook into different parts of the time-stepping cycle.
For example, use `callsite = TendencyCallsite()` to modify tendencies before a step, or
`UpdateStateCallsite()` to react after auxiliary variables update. See the [Callbacks page](@ref callbacks)
for more customization patterns, including parameterized callbacks and state callbacks.

```@example checkpointing
using Oceananigans
using Oceananigans: TendencyCallsite

model = NonhydrostaticModel(RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))
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

## Introduction to writing output

Output writers live in the ordered dictionary `simulation.output_writers`. Each writer pairs
outputs with a schedule describing _when_ to export them and a backend (NetCDF, JLD2, or
Checkpointer) describing _how_ to serialize them. Time-step alignment ensures the writer's
schedule is honored without user intervention.

```@example simulation_overview
using NCDatasets

fields = Dict("u" => simulation.model.velocities.u)

simulation.output_writers[:surface_slice] = NetCDFWriter(simulation.model, fields;
                                                         filename = "demo.nc",
                                                         schedule = TimeInterval(30),
                                                         indices = (:, :, grid.Nz))
simulation.output_writers
```

During `run!`, Oceananigans calls each writer whenever its schedule actuates.
More comprehensive information may be found on the [output writers page](@ref output_writers).

## Putting it together

Combining callbacks, output writers, and adaptive time-stepping turns a few lines of model code
into a robust workflow:

```@example simulation_overview
using Oceananigans.Units: hours, minutes

model = NonhydrostaticModel(grid; tracers=:T)
simulation = Simulation(model, Δt=20, stop_time=2hours)

progress(sim) = @info "t = $(prettytime(sim)), Δt = $(prettytime(sim.Δt))"
add_callback!(simulation, progress, IterationInterval(10))
conjure_time_step_wizard!(simulation, cfl=0.8)

simulation.output_writers[:snapshots] = JLD2Writer(simulation.model, simulation.model.velocities;
                                                   filename = "snapshots.jld2",
                                                   schedule = TimeInterval(30minutes))

run!(simulation)
```

For more recipes continue with the page on [schedules](@ref schedules).
