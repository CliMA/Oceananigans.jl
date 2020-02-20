# Diagnostics
Diagnostics are a set of general utilities that can be called on-demand during time-stepping to compute quantities of
interest you may want to save to disk, such as the horizontal average of the temperature, the maximum velocity, or to
produce a time series of salinity. They also include utilities for diagnosing model health, such as the CFL number or
to check for NaNs.

Diagnostics are stored as a list of diagnostics in `simulation.diagnostics`. Diagnostics can be specified at model creation
time or be specified at any later time and appended (or assigned with a key value pair) to `simulation.diagnostics`.

Most diagnostics can be run at specified frequencies (e.g. every 25 time steps) or specified intervals (e.g. every
15 minutes of simulation time). If you'd like to run a diagnostic on demand then do not specify a frequency or interval
(and do not add it to `simulation.diagnostics`).

We describe the `HorizontalAverage` diagnostic in detail below but see the API documentation for other diagnostics such
as [`TimeSeries`](@ref), [`FieldMaximum`](@ref), [`CFL`](@ref), and [`NaNChecker`](@ref).

## Horizontal averages
You can create a `HorizontalAverage` diagnostic by passing a field to the constructor, e.g.
```@example
model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))
simulation = Simulation(model, Δt=6, stop_iteration=10)
T_avg = HorizontalAverage(model.tracers.T)
push!(simulation.diagnostics, T_avg)
```
which can then be called on-demand via `T_avg(model)` to return the horizontally averaged temperature. When running on
the GPU you may want it to return an `Array` instead of a `CuArray` in case you want to save the horizontal average to
disk in which case you'd want to construct it like
```@example
model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))
simulation = Simulation(model, Δt=6, stop_iteration=10)
T_avg = HorizontalAverage(model.tracers.T, return_type=Array)
push!(simulation.diagnostics, T_avg)
```

You can also use pass an abstract operator to take the horizontal average of any diagnosed quantity. For example, to
compute the horizontal average of the vertical component of vorticity:
```@example
model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))
simulation = Simulation(model, Δt=6, stop_iteration=10)
u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)
ζ_avg = HorizontalAverage(ζ)
simulation.diagnostics[:vorticity_profile] = ζ_avg
```

See [`HorizontalAverage`](@ref) for more details and options.
