# Quick start

Try copy/pasting this into a Julia REPL:

```jldoctest cpu
using Oceananigans
grid = RectilinearGrid(size=(128, 128), halo=(3, 3), x=(0, 2π), y=(0, 2π), topology=(Periodic, Periodic, Flat))
model = NonhydrostaticModel(; grid, advection=WENO5())
ϵ(x, y, z) = 2rand() - 1
set!(model, u=ϵ, v=ϵ)
simulation = Simulation(model; Δt=0.01, stop_iteration=100)
run!(simulation)
```

But you want to run on your super fancy GPU? Well for that you write,

```jldoctest gpu
using Oceananigans
grid = RectilinearGrid(GPU(), size=(128, 128), halo=(3, 3), x=(0, 2π), y=(0, 2π), topology=(Periodic, Periodic, Flat))
model = NonhydrostaticModel(; grid, advection=WENO5())
ϵ(x, y, z) = 2rand() - 1
set!(model, u=ϵ, v=ϵ)
simulation = Simulation(model; Δt=0.01, stop_iteration=100)
run!(simulation)
```

## Visualization

Visualization in Julia...

```jldoctest cpu
using GLMakie

u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))
compute!(ζ)
heatmap(interior(ζ, :, :, 1))
```
