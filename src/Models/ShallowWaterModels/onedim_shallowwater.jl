# # One dimensional ShallowWater example
#
# ## Model setup

using Oceananigans, Oceananigans.Advection

include("ShallowWaterModels.jl")
using .ShallowWaterModels: ShallowWaterModel

grid = RegularCartesianGrid(size=(1, 1, 1), extent=(2π, 2π, 2π))

model = ShallowWaterModel(        grid = grid,
                          architecture = CPU(),
                             advection = UpwindBiasedFifthOrder(),
                              coriolis = FPlane(f=0.0),
                            velocities = nothing,
                               tracers = nothing,
                           layer_depth = nothing,
                           timestepper = :RungeKutta3 
                           )

width = 0.1

layer_depth(x, y, z) = exp(-x^2 / (2width^2)) 
          u(x, y, z) = 0.0
          v(x, y, z) = 0.0

#set!(model, u = u, v = v)
set!(model, u = u, v = v, layer_deth = layer_depth)
#fails with "ERROR: LoadError: ArgumentError: name layer_depth not found in model.velocities or model.tracers."



