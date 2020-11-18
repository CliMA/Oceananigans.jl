module ShallowWaterModels

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!

#####
##### ShallowWaterModel definition
#####

include("shallow_water_model.jl")

#####
##### Time-stepping ShallowWaterModels
#####

end 
