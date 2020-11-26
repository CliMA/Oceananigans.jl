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
include("show_shallow_water_model.jl")

#####
##### Time-stepping ShallowWaterModels
#####

"""
    fields(model::ShallowWaterModel)

Returns a flattened `NamedTuple` of the fields in `model.solution` and `model.tracers`.
"""
fields(model::ShallowWaterModel) = merge(model.solution, model.tracers)

include("solution_and_tracer_tendencies.jl")
include("calculate_shallow_water_tendencies.jl")
include("update_shallow_water_state.jl")

end # module
