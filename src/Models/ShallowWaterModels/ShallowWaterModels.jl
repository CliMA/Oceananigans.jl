module ShallowWaterModels

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!

#####
##### ShallowWaterModel definition
#####

include("shallowwater_model.jl")

#####
##### Time-stepping ShallowWaterModels
#####

end 
