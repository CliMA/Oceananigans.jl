# # One dimensional shallow water example

using Oceananigans, Oceananigans.Advection

include("../src/Models/ShallowWaterModels/ShallowWaterModels.jl")

using .ShallowWaterModels: ShallowWaterModel

include("../src/Models/ShallowWaterModels/set_shallow_water_model.jl")
include("../src/Models/ShallowWaterModels/update_shallow_water_state.jl")
include("../src/Models/ShallowWaterModels/calculate_shallow_water_tendencies.jl")
include("../src/Models/ShallowWaterModels/solution_and_tracer_tendencies.jl")

grid = RegularCartesianGrid(size=(64, 1, 1), extent=(2π, 2π, 2π))

model = ShallowWaterModel(        grid = grid,
                          architecture = CPU(),
                             advection = nothing, 
                              coriolis = nothing
#                             solution = nothing,
#                              tracers = nothing
                                  )

width = 0.3
 h(x, y, z)  = 1.0 + 0.1 * exp(-(x - π)^2 / (2width^2));  
uh(x, y, z) = 0.0
vh(x, y, z) = 0.0 

set!(model, uh = uh, vh = vh, h = h)

simulation = Simulation(model, Δt = 0.1, stop_iteration = 10)

run!(simulation)
