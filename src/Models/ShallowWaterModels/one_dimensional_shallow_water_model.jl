# # One dimensional shallow water example

using Oceananigans, Oceananigans.Advection

include("ShallowWaterModels.jl")
using .ShallowWaterModels: ShallowWaterModel

grid = RegularCartesianGrid(size=(64, 1, 1), extent=(2π, 2π, 2π))

model = ShallowWaterModel(        grid = grid,
                          architecture = CPU(),
                             advection = nothing, 
                              coriolis = nothing, 
                             solution = nothing,
                              tracers = nothing,
                           timestepper = :RungeKutta3 
                                  )

width = 0.3

 h(x, y, z)  = exp(-x^2 / (2width^2)) 
uh(x, y, z) = 0.0
vh(x, y, z) = 0.0


include("set_shallow_water_model.jl")
set!(model, uh = uh, vh = vh, h = h)

using Plots
using Oceananigans.Grids: xnodes 

x = xnodes(model.solution.h)

h_plot = plot(x, interior(model.solution.h)[:, 1, 1],
              linewidth = 2,
              label = "t = 0",
              xlabel = "height (m)",
              ylabel = "x")

#=
#progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(round(Int, sim.model.clock.time))"

#simulation = Simulation(model, Δt=0.1, stop_time=10, iteration_interval=10, progress=progress)

#run!(simulation)
 =#
