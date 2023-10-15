# Quick start

This code:

```@example cpu
using Oceananigans

grid = RectilinearGrid(size=(128, 128), x=(0, 2π), y=(0, 2π), topology=(Periodic, Periodic, Flat))
model = NonhydrostaticModel(; grid, advection=WENO())

ϵ(x, y, z) = 2rand() - 1
set!(model, u=ϵ, v=ϵ)

simulation = Simulation(model; Δt=0.01, stop_iteration=100)
run!(simulation)
```

runs 100 time steps of a two-dimensional turbulence simulation with `128²` [finite volume](https://en.wikipedia.org/wiki/Finite_volume_method) cells and a fifth-order upwinded [WENO advection scheme](https://en.wikipedia.org/wiki/WENO_methods).
It's quite similar to the [two-dimensional turbulence example](https://clima.github.io/OceananigansDocumentation/stable/generated/two_dimensional_turbulence/).

## Visualization

They say that a [Makie](https://makie.juliaplots.org/stable/) visualization is worth a thousand Unicode characters, so let's plot vorticity,

```@example cpu
using CairoMakie

u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))
compute!(ζ)

heatmap(interior(ζ, :, :, 1))
```

A few more time-steps, and it's starting to get a little diffuse!

```@example cpu
simulation.stop_iteration += 400
run!(simulation)

compute!(ζ)
heatmap(interior(ζ, :, :, 1))
```

## They always cheat with too-simple "quick" starts

Fine, we'll re-run this code on the GPU then:

```julia
using Oceananigans

grid = RectilinearGrid(GPU(), size=(128, 128), x=(0, 2π), y=(0, 2π), topology=(Periodic, Periodic, Flat))
model = NonhydrostaticModel(; grid, advection=WENO())

ϵ(x, y, z) = 2rand() - 1
set!(model, u=ϵ, v=ϵ)

simulation = Simulation(model; Δt=0.01, stop_iteration=100)
run!(simulation)
```

Notice the difference? We passed the positional argument `GPU()` to `RectilinearGrid`.

## Well, that was tantalizing

But you'll need to know a lot more to become a productive, Oceananigans-wielding computational scientist (spherical grids, forcing, boundary conditions, turbulence closures, output writing, actually labeling your axes... 🤯). It'd be best to move on to the [one-dimensional diffusion example](@ref one_dimensional_diffusion_example).
