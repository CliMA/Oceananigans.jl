# # One dimensional ShallowWater example
#
# ## Model setup

using Oceananigans, Oceananigans.Advection

include("ShallowWaterModels.jl")
using .ShallowWaterModels: ShallowWaterModel

grid = RegularCartesianGrid(size=(1, 1, 1), extent=(2π, 2π, 2π))

# When I try this in REPL I get a lot of stuff in the output that I should't.  Why?
model = ShallowWaterModel(        grid = grid,
                          architecture = CPU(),
                             advection = nothing, 
                              coriolis = nothing, 
                            velocities = nothing,
                               tracers = (:D),
                           timestepper = :RungeKutta3 
                           )

width = 0.1

D(x, y, z) = exp(-x^2 / (2width^2)) 
u(x, y, z) = 0.0
v(x, y, z) = 0.0

set!(model, u = u, v = v, D = D)

progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(round(Int, sim.model.clock.time))"

simulation = Simulation(model, Δt=0.1, stop_time=10, iteration_interval=10, progress=progress)

run!(simulation)
