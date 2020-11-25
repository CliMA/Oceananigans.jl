module ShallowWaterModels

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!

import Oceananigans: fields

#####
##### ShallowWaterModel definition
#####

include("shallow_water_model.jl")
include("set_shallow_water_model.jl")

#####
##### Time-stepping ShallowWaterModels
#####

fields(model::ShallowWaterModel) = merge(model.solution, model.tracers)

end 
