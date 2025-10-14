# [Scheduling callbacks and output writers](@id callback_schedules)

Callbacks and output writers in `Simulation` actuate on objects that subtype
[`Oceananigans.Utils.AbstractSchedule`](@ref Oceananigans.Utils.AbstractSchedule).
Schedules are small callable objects that return `true` when an action should fire and `false` otherwise.
This page collects the built-in schedules, when to use them, and how to combine them for more complex behavior.

!!! tip "Aligned time steps"
    `Simulation` automatically shortens the next time step so that callback and output events scheduled by model
    time happen exactly when requested. Set `align_time_step = false` in the `Simulation` constructor to disable this.

## Core schedules

The most common constructors live in `Oceananigans.Utils` and work for both callbacks and output writers.

### [`TimeInterval`](@ref)

`TimeInterval(interval)` actuates every `interval` seconds of model time. The constructor accepts numbers or units
from `Oceananigans.Units`.

```@example
using Oceananigans
using Oceananigans.Units: minutes
TimeInterval(30minutes)
```

### [`IterationInterval`](@ref)

`IterationInterval(interval)` actuates every `interval` iterations:

```@example
using Oceananigans
IterationInterval(11)
```

Use the `offset` kwarg to shift the trigger so that, for example,

```@example
IterationInterval(100; offset=-1)
```

runs on iterations 99, 199, 299, …

### [`WallTimeInterval`](@ref)

`WallTimeInterval(interval; start_time=time_ns()*1e-9)` uses wall-clock seconds instead of model time. This is useful
for long simulations where you want periodic status updates regardless of adaptive time-stepping. Because it depends
on real time, it may produce uneven iteration spacing on faster or slower hardware.

### [`SpecifiedTimes`](@ref)

`SpecifiedTimes(times...)` actuates when `model.clock.time` reaches the given values. The constructor accepts numeric
times or `Dates.DateTime` values and sorts them automatically. This schedule is helpful for pre-planned save points or
events tied to specific model times.

## Combining schedules

Some applications benefit from running extra steps immediately after an event or from combining multiple criteria.

### [`ConsecutiveIterations`](@ref Oceananigans.Utils.ConsecutiveIterations)

`ConsecutiveIterations(parent_schedule, N=1)` actuates when the parent schedule does and for the next `N` iterations.
For example, averaging callbacks often need data at the scheduled time and immediately afterwards.

### [`AndSchedule`](@ref) and [`OrSchedule`](@ref)

Use `AndSchedule(s₁, s₂, ...)` when an action should fire only if every child schedule actuates in the same iteration.
Use `OrSchedule(s₁, s₂, ...)` when any one of the child schedules should trigger the action. Both accept any mix of
`AbstractSchedule`s, so you can require, for example, output every hour *and* every 1000 iterations:

```julia
combo = AndSchedule(TimeInterval(1hour), IterationInterval(1000))
simulation.callbacks[:hourly_checkpoint] = Callback(stop_simulation, combo)
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

## Writing custom schedules

Any callable object or function that accepts a model and returns `true` / `false` can be used as a schedule. Implement
`Oceananigans.initialize!(schedule, model)` if the schedule needs to reset internal state when attached to different
callbacks or output writers.

For deeper examples of attaching schedules see the [callbacks tutorial](@ref callbacks) and the
[output writers guide](@ref output_writers).
