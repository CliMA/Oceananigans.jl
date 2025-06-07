using Oceananigans
grid = RectilinearGrid(topology = (Bounded, Flat, Bounded), size=(4, 4), extent=(1, 1))
model = NonhydrostaticModel(; grid)
time_step!(model, 1)
simulation = Simulation(model; Δt = 1, stop_time=10)
run!(simulation)
