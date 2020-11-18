# # One dimensional ShallowWater example
#
# ## Model setup

using Oceananigans, Oceananigans.Advection

include("ShallowWaterModels.jl")
using .ShallowWaterModels: ShallowWaterModel

grid = RegularCartesianGrid(size=(32, 1, 1), extent=(2π, 2π, 2π))

# When I try this in REPL I get a lot of stuff in the output that I should't.  Why?
model = ShallowWaterModel(        grid = grid,
                          architecture = CPU(),
                             advection = nothing, 
                              coriolis = nothing, 
                            transports = nothing,
                               heights = nothing,
                               tracers = nothing,
                           timestepper = :RungeKutta3 
                           )

width = 0.1

h(x, y)  = exp(-x^2 / (2width^2)) 
uh(x, y) = 0.0
vh(x, y) = 0.0

include("../../Fields/set_new!.jl")
set_new!(model, uh = uh, vh = vh, h = h)

#=
using Plots
using Oceananigans.Grids: xnodes 

x = xnodes(model.tracers.D)

D_plot = plot(interior(model.tracers.D)[:, 1, 1], x,
              linewidth = 2,
              label = "t = 0",
              xlabel = "Depth (m)",
              ylabel = "x")

#progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(round(Int, sim.model.clock.time))"

#simulation = Simulation(model, Δt=0.1, stop_time=10, iteration_interval=10, progress=progress)

#run!(simulation)
 =#
