using Oceananigans

grid_base = RectilinearGrid(topology = (Bounded, Periodic, Bounded), size = (16, 20, 4), extent = (800, 1000, 100),)
    
@inline east_wall(x, y, z) = x > 400
grid = ImmersedBoundaryGrid(grid_base, GridFittedBoundary(east_wall))
model = NonhydrostaticModel(grid = grid, timestepper = :RungeKutta3, buoyancy = BuoyancyTracer(), tracers = :b, hydrostatic_pressure_anomaly=nothing)

N² = 6e-6
b∞(x, y, z) = N² * z
set!(model, b=b∞)
    
simulation = Simulation(model, Δt=25, stop_time=1e4,)

using Statistics: std
using Printf
progress_message(sim) = @printf("Iteration: %04d, time: %s, iteration×Δt: %s, std(pNHS) = %.2e\n",
                                iteration(sim), sim.model.clock.time, iteration(sim) * sim.Δt, std(model.pressures.pNHS))
add_callback!(simulation, progress_message, TimeInterval(100))

run!(simulation)
