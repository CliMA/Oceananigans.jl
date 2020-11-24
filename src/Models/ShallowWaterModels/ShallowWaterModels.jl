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

# These files can be removed when rk3_substep! and store_tendencies! are generalized:
include("rk3_substep_shallow_water_model.jl")
include("store_shallow_water_tendencies.jl")

end 
