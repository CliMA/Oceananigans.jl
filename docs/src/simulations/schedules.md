# [Scheduling callbacks and output writers](@id callback_schedules)

Callbacks and output writers in `Simulation` actuate on objects that subtype
[`Oceananigans.Utils.AbstractSchedule`](@ref Oceananigans.Utils.AbstractSchedule).
Schedules are small callable objects that return `true` when an action should fire and `false` otherwise.
This page collects the built-in schedules, when to use them, and how to combine them for more complex behavior.

!!! tip "Aligned time steps"
    `Simulation` automatically shortens the next time step so that callback and output events scheduled by model
    time happen exactly when requested. Set `align_time_step = false` in the `Simulation` constructor to disable this.

For the following examples, we will use the following simple simulation and progress:

```@example schedules
using Oceananigans
grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = NonhydrostaticModel(; grid)
simulation = Simulation(model, Δt=0.1, stop_time=2.5, verbose=false)
dummy(sim) = @info string("Iter: ", iteration(sim), "-- I was called at t = ", time(sim))
```

## [`IterationInterval`](@ref)

`IterationInterval(interval)` actuates every `interval` iterations:

```@example schedules
schedule = IterationInterval(11)
add_callback!(simulation, dummy, schedule, name=:dummy)
run!(simulation)
```

Use the `offset` kwarg to shift the trigger so that, for example,

```@example schedules
Oceananigans.Simulations.reset!(simulation)
simulation.stop_time = 2.5

schedule = IterationInterval(7; offset=-2)
add_callback!(simulation, dummy, schedule, name=:dummy)
run!(simulation)
```

notice that the callback is actuated on iterations 5, 12, 19, …

## [`TimeInterval`](@ref)

`TimeInterval(interval)` actuates every `interval` of model time,
in units corresponding to `model.clock.time`. For example,

```@example schedules
Oceananigans.Simulations.reset!(simulation)
simulation.stop_time = 2.5

schedule = TimeInterval(1.11)
add_callback!(simulation, dummy, schedule, name=:dummy)
run!(simulation)
```

When `model.clock.time isa AbstractTime` such as `DateTime`, then `interval` can be `Dates.Period`:

```@example schedules
using Dates

start_time = DateTime(2025, 1, 1)
clock = Clock(time = start_time)
datetime_model = NonhydrostaticModel(; grid, clock)

stop_time = start_time + Dates.Minute(3)
datetime_simulation = Simulation(datetime_model; Δt=Dates.Second(25), stop_time, verbose=false)

schedule = TimeInterval(Dates.Minute(1))
add_callback!(datetime_simulation, dummy, schedule)
run!(datetime_simulation)
```

If `interval isa Number` with an `AbstractTime` clock, then `interval`
is interpreted as a `Dates.Second`:

```@example schedules
datetime_model = NonhydrostaticModel(; grid, clock, timestepper = :QuasiAdamsBashforth2)
stop_time = start_time + Dates.Minute(3)
datetime_simulation = Simulation(datetime_model; Δt=Dates.Second(25), stop_time, verbose=false)

schedule = TimeInterval(59)
add_callback!(datetime_simulation, dummy, schedule, name=:dummy)
run!(datetime_simulation)
```

### [`WallTimeInterval`](@ref)

`WallTimeInterval(interval; start_time=time_ns()*1e-9)` uses wall-clock seconds instead of model time.
This is mostly useful for writing checkpoints to disk after consuming a fixed amount of computational resources.
For example,

```@example schedules
Oceananigans.Simulations.reset!(simulation)
simulation.stop_time = 2.5

schedule = WallTimeInterval(1e-1)
add_callback!(simulation, dummy, schedule, name=:dummy)
run!(simulation)
```

### [`SpecifiedTimes`](@ref)

`SpecifiedTimes(times...)` actuates when `model.clock.time` reaches the given values.
The constructor accepts numeric times or `Dates.DateTime` values and sorts them automatically.
This schedule is helpful for pre-planned save points or events tied to specific model times.

```@example schedules
Oceananigans.Simulations.reset!(simulation)
simulation.stop_time = 2.5

schedule = SpecifiedTimes(0.2, 1.5, 2.1)
add_callback!(simulation, dummy, schedule, name=:dummy)
run!(simulation)
```

## Arbitrary functions of model

Any function of `model` that returns a `Bool` can be used as a schedule:

```@example schedules
Oceananigans.Simulations.reset!(simulation)
simulation.stop_time = 2.5

after_two(model) = model.clock.time > 2
add_callback!(simulation, dummy, after_two, name=:dummy)
run!(simulation)
```

## Combining schedules

Some applications benefit from running extra steps immediately after an event or from combining multiple criteria.

### [`ConsecutiveIterations`](@ref Oceananigans.Utils.ConsecutiveIterations)

`ConsecutiveIterations(parent_schedule, N=1)` actuates when the parent schedule does and for the next `N` iterations.
For example, averaging callbacks often need data at the scheduled time and immediately afterwards.

```@example schedules
Oceananigans.Simulations.reset!(simulation)
simulation.stop_time = 2.5

times = SpecifiedTimes(0.2, 1.5, 2.1)
schedule = ConsecutiveIterations(times)
add_callback!(simulation, dummy, schedule, name=:dummy)
run!(simulation)
```

### [`AndSchedule`](@ref) and [`OrSchedule`](@ref)

Use `AndSchedule(s₁, s₂, ...)` when an action should fire only if every child schedule actuates in the same iteration.
Use `OrSchedule(s₁, s₂, ...)` when any one of the child schedules should trigger the action. Both accept any mix of
`AbstractSchedule`s, so you can require, for example, output every hour *and* every 1000 iterations:

```@example schedules
Oceananigans.Simulations.reset!(simulation)
simulation.stop_time = 2.5

after_one_point_seven(model) = model.clock.time > 1.7
schedule = AndSchedule(IterationInterval(2), after_one_point_seven)
add_callback!(simulation, dummy, schedule, name=:dummy)
run!(simulation)
```

Stateful schedules such as `TimeInterval`, `SpecifiedTimes`, and `ConsecutiveIterations`
store their own counters, so create a fresh instance (or call `copy`) for each callback or output writer that needs an identical pattern.

## Output-specific schedules

Some schedules only apply to output writers because they keep extra state or require file access.

### [`AveragedTimeInterval`](@ref)

`AveragedTimeInterval(interval; window=interval, stride=1)` asks an output writer to accumulate data over a sliding time
window before writing. The window ends at each actuation time, runs for `window` seconds, and samples every `stride`
iterations inside the window.

### [`AveragedSpecifiedTimes`](@ref Oceananigans.OutputWriters.AveragedSpecifiedTimes)

`AveragedSpecifiedTimes(times; window, stride=1)` behaves like `SpecifiedTimes` but with a trailing averaging window.
Pass either a `SpecifiedTimes` instance or raw times.

### [`FileSizeLimit`](@ref)

`FileSizeLimit(size_limit)` actuates when the target file grows beyond `size_limit` bytes. Output writers update the
internal path automatically so you usually only pass the size limit. Combine it with `OrSchedule` to rotate files when
either the clock reaches a value or the file becomes too large.
