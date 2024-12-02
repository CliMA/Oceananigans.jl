using Oceananigans
using GLMakie

grid = RectilinearGrid(size=(128, 32), x=(-5, 5), z=(0, 1), topology=(Periodic, Flat, Bounded))

mountain(x) = 0.1 * exp(-x^2)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(mountain))

Fu(x, z, t) = sin(t)
model = NonhydrostaticModel(; grid, advection=WENO(order=5), forcing=(; u=Fu))
simulation = Simulation(model, Δt=1e-3, stop_iteration=100)
run!(simulation)
heatmap(model.velocities.u)
fig = current_figure()

