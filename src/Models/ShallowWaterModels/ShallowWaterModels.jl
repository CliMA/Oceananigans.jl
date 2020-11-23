module ShallowWaterModels

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!

import Oceananigans.Models: fields

#####
##### ShallowWaterModel definition
#####

include("shallow_water_model.jl")
include("set_shallow_water_model.jl")

#####
##### Time-stepping ShallowWaterModels
#####

"""
    fields(model::ShallowWaterModel)
Returns a flattened `NamedTuple` of the fields in `model.solution`.
"""
fields(model::ShallowWaterModel) = merge(model.solution, model.tracers)

include("calculate_shallow_water_tendencies.jl")

end 
