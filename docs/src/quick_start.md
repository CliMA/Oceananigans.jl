# [Quick start](@id quick_start)

This code:

```@setup cpu
using CairoMakie
CairoMakie.activate!(type = "png")
```

```@example cpu
using Oceananigans

grid = RectilinearGrid(size = (128, 128),
                       x = (0, 2π),
                       y = (0, 2π),
                       topology = (Periodic, Periodic, Flat))

model = NonhydrostaticModel(grid; advection=WENO())

ϵ(x, y) = 2rand() - 1
set!(model, u=ϵ, v=ϵ)

simulation = Simulation(model; Δt=0.01, stop_iteration=100)
run!(simulation)
```

runs 100 time steps of a two-dimensional turbulence simulation with `128²` [finite volume](https://en.wikipedia.org/wiki/Finite_volume_method) cells
and a fifth-order upwinded [WENO advection scheme](https://en.wikipedia.org/wiki/WENO_methods).
It's quite similar to the [two-dimensional turbulence example](@ref two_dimensional_turbulence).

## Visualization

They say that a [Makie](https://makie.juliaplots.org/stable/) visualization is worth a thousand Unicode characters, so let's plot vorticity,

```@example cpu
using CairoMakie

u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))

heatmap(ζ, axis=(; aspect=1))
```

A few more time-steps, and it's starting to get a little diffuse!

```@example cpu
simulation.stop_iteration += 400
run!(simulation)

heatmap(ζ, axis=(; aspect=1))
```

## They always cheat with too-simple "quick" starts

Fine, we'll re-run this code on the GPU. But we're a little greedy, so we'll also
crank up the resolution, throw in a `TimeStepWizard` to update `simulation.Δt` adaptively,
and add a passive tracer initially concentrated in the center of the domain
which will make for an even prettier figure of the final state:

```@setup gpu
using CairoMakie
CairoMakie.activate!(type = "png")
```

```@example gpu
using Oceananigans
using CairoMakie
using CUDA

grid = RectilinearGrid(GPU(),
                       size = (1024, 1024),
                       x = (-π, π),
                       y = (-π, π),
                       topology = (Periodic, Periodic, Flat))

model = NonhydrostaticModel(grid; advection=WENO(), tracers=:c)

δ = 0.5
cᵢ(x, y) = exp(-(x^2 + y^2) / 2δ^2)
ϵ(x, y) = 2rand() - 1
set!(model, u=ϵ, v=ϵ, c=cᵢ)

simulation = Simulation(model; Δt=1e-3, stop_time=10)
conjure_time_step_wizard!(simulation, cfl=0.2, IterationInterval(10))
run!(simulation)

u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))

fig = Figure(size=(1200, 600))
axζ = Axis(fig[1, 1], aspect=1, title="vorticity")
axc = Axis(fig[1, 2], aspect=1, title="tracer")
heatmap!(axζ, ζ, colormap=:balance)
heatmap!(axc, model.tracers.c)
current_figure()
```

See how we did that? We passed the positional argument `GPU()` to `RectilinearGrid`.
(This only works if a GPU is available, of course, and
[CUDA.jl is configured](https://cuda.juliagpu.org/stable/installation/overview/).)

## Units

`Oceananigans.Units` provides `Float64` constants for expressing physical quantities
as plain numeric products. These are not special types, just multiplication to convert to SI units:

```julia
using Oceananigans.Units

Δt = 5minutes      # = 300.0
stop_time = 10days # = 864000.0
Lx = 500kilometers # = 500000.0
```

Available units:

| Quantity | Constants | Value (SI) |
|----------|-----------|------------|
| Time     | `second`, `seconds` | 1 s |
| Time     | `minute`, `minutes` | 60 s |
| Time     | `hour`, `hours`     | 3600 s |
| Time     | `day`, `days`       | 86400 s |
| Length   | `meter`, `meters`   | 1 m |
| Length   | `kilometer`, `kilometers` | 1000 m |

Singular and plural forms are identical (`1day == 1days`).
The file-size constants `KiB`, `MiB`, `GiB`, and `TiB` are also available (for use with output writers).

## Well, that was tantalizing

But you'll need to know a lot more to become a productive, Oceananigans-wielding computational scientist (spherical grids, forcing, boundary conditions,
turbulence closures, output writing, actually labeling your axes... 🤯).
It'd be best to move on to the [one-dimensional diffusion example](@ref one_dimensional_diffusion_example).
