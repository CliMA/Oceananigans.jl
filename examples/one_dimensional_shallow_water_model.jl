# # One dimensional shallow water example

using Oceananigans, Oceananigans.Advection

include("../src/Models/ShallowWaterModels/ShallowWaterModels.jl")
using .ShallowWaterModels: ShallowWaterModel

grid = RegularCartesianGrid(size=(64, 1, 1), extent=(2π, 2π, 2π))

model = ShallowWaterModel(        grid = grid,
                          architecture = CPU(),
                             advection = nothing, 
                              coriolis = nothing, 
                             solution = nothing,
                              tracers = nothing#,
#                          timestepper = RungeKutta3 
                                  )

width = 0.3
 h(x, y, z)  = 1.0 + 0.1 * exp(-(x - π)^2 / (2width^2)); 
uh(x, y, z) = 0.0
vh(x, y, z) = 0.0

include("../src/Models/ShallowWaterModels/set_shallow_water_model.jl")
set!(model, uh = uh, vh = vh, h = h)

using Plots
using Oceananigans.Grids: xnodes 

x = xnodes(model.solution.h)

h_plot = plot(x, interior(model.solution.h)[:, 1, 1],
              linewidth = 3,
                  label = "t = 0",
                 xlabel = "x (m)",
                 ylabel = "",
                 title  = "Height (m)")

display(h_plot)
savefig("initial_height.png")

simulation = Simulation(model, Δt = 0.1, stop_iteration = 10)

include("/home/fpoulin/software/Oceananigans.jl/src/Models/ShallowWaterModels/update_state.jl")
include("/home/fpoulin/software/Oceananigans.jl/src/Models/ShallowWaterModels/calculate_tendencies.jl")
include("/home/fpoulin/software/Oceananigans.jl/src/Models/ShallowWaterModels/solution_and_tracer_tendencies.jl")

run!(simulation)
