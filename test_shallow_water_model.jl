using Oceananigans.Architectures: CPU
using Oceananigans.Grids: Periodic, Bounded, RegularCartesianGrid
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Simulations: Simulation, set!, run!

grid = RegularCartesianGrid(size=(1, 1, 1), extent=(2π, 2π, 2π))
model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=CPU())
set!(model, h=1)

simulation = Simulation(model, Δt=1.0, stop_iteration=1)
run!(simulation)