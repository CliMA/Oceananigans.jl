# Diagnostics

Diagnostics are a set of general utilities that can be called on-demand during time-stepping to compute quantities of
interest you may want to save to disk, such as the horizontal average of the temperature, the maximum velocity, or to
produce a time series of salinity. They also include utilities for diagnosing model health, such as the CFL number or
to check for NaNs.

Diagnostics are stored as a list of diagnostics in `simulation.diagnostics`. Diagnostics can be specified at model creation
time or be specified at any later time and appended (or assigned with a key value pair) to `simulation.diagnostics`.

Most diagnostics can be run at specified frequencies (e.g. every 25 time steps) or specified intervals (e.g. every
15 minutes of simulation time). If you'd like to run a diagnostic on demand then do not specify any intervals
(and do not add it to `simulation.diagnostics`).

We describe the [`Average`](@ref) diagnostic in detail below but see the API documentation for other diagnostics such
as [`TimeSeries`](@ref), [`FieldMaximum`](@ref), [`CFL`](@ref), and [`NaNChecker`](@ref).

## Horizontal averages

You can create a horizontal `Average` diagnostic by passing a field to the constructor, e.g.

```@meta
DocTestSetup = quote
    using Oceananigans
    using Oceananigans.Diagnostics
end
```

```jldoctest
julia> model = IncompressibleModel(grid=RegularCartesianGrid(size=(4, 4, 4), extent=(1, 1, 1)));

julia> T_avg = AveragedField(model.tracers.T, dims=(1, 2));

julia> T_avg(model)  # Compute horizontal average of T on demand
1×1×6 Array{Float64,3}:
[:, :, 1] =
 0.0

[:, :, 2] =
 0.0

[:, :, 3] =
 0.0

[:, :, 4] =
 0.0

[:, :, 5] =
 0.0

[:, :, 6] =
 0.0
```

which can then be called on-demand via `T_avg(model)` to return the horizontally averaged temperature. Notice that
halo regions are included in the output of the horizontal average. When running on the GPU you may want it to return
an `Array` instead of a `CuArray` in case you want to save the horizontal average to disk in which case you'd want to
construct it like

```jldoctest
julia> model = IncompressibleModel(grid=RegularCartesianGrid(size=(4, 4, 4), extent=(1, 1, 1)));

julia> T_avg = AveragedField(model.tracers.T, dims=(1, 2), return_type=Array);

julia> T_avg(model)  # Will always return an Array
1×1×6 Array{Float64,3}:
[:, :, 1] =
 0.0

[:, :, 2] =
 0.0

[:, :, 3] =
 0.0

[:, :, 4] =
 0.0

[:, :, 5] =
 0.0

[:, :, 6] =
 0.0
```

You can also use pass an abstract operator to take the horizontal average of any diagnosed quantity. For example, to
compute the horizontal average of the vertical component of vorticity:

```julia
model = IncompressibleModel(grid=RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1)))
simulation = Simulation(model, Δt=6, stop_iteration=10)

u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)
ζ_avg = AveragedField(ζ, dims=(1, 2))
simulation.diagnostics[:vorticity_profile] = ζ_avg
```

See [`Average`](@ref) for more details and options.
