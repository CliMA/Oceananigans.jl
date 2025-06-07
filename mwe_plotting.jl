using Oceananigans

underlying_grid = RectilinearGrid(topology = (Bounded, Flat, Bounded), size = (4, 4), extent = (1, 1))
bottom(x) = -1/2
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

model = NonhydrostaticModel(; grid,)
simulation = Simulation(model; Δt=1, stop_iteration=10)

using CairoMakie
a = heatmap(model.velocities.u)
run!(simulation)
