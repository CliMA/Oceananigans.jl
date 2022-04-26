# Quick start

This code:

```jldoctest cpu
using Oceananigans
grid = RectilinearGrid(size=(128, 128), halo=(3, 3), x=(0, 2π), y=(0, 2π), topology=(Periodic, Periodic, Flat))
model = NonhydrostaticModel(; grid, advection=WENO5())
ϵ(x, y, z) = 2rand() - 1
set!(model, u=ϵ, v=ϵ)
simulation = Simulation(model; Δt=0.01, stop_iteration=100)
run!(simulation)
```

runs 100 time steps of a two-dimensional turbulence simulation with `128²` finite volume cells and a fifth-order upwinded WENO advection scheme.
It's quite similar to the [two-dimensional turbulence example](https://clima.github.io/OceananigansDocumentation/stable/generated/two_dimensional_turbulence/).

## Visualization

They say that a Makie visualization is worth a thousand unicode characters, so here's a plot of vorticity...

```jldoctest cpu
using GLMakie

u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))
compute!(ζ)
heatmap(interior(ζ, :, :, 1))
```

## GPU quick start

Switching over to the GPU requires changing just a few characters (at least, for this simple setup):

```jldoctest gpu
using Oceananigans
grid = RectilinearGrid(GPU(), size=(128, 128), halo=(3, 3), x=(0, 2π), y=(0, 2π), topology=(Periodic, Periodic, Flat))
model = NonhydrostaticModel(; grid, advection=WENO5())
ϵ(x, y, z) = 2rand() - 1
set!(model, u=ϵ, v=ϵ)
simulation = Simulation(model; Δt=0.01, stop_iteration=100)
run!(simulation)
```

Notice the difference? We passed the positional argument `GPU()` to `RectilinearGrid`.

## Well, that was tantalizing

Of course, you'll need to know a lot more to become a productive, Oceananigans-wielding computational scientist (spherical grids, forcing, boundary conditions, turbulence closures, output writing... 🤯). It'd be best to move on to the [one-dimensional diffusion example](https://clima.github.io/OceananigansDocumentation/stable/generated/one_dimensional_diffusion/).
