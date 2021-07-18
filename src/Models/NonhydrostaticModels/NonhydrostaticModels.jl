module NonhydrostaticModels

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!

import Oceananigans: fields, prognostic_fields

#####
##### NonhydrostaticModel definition
#####

include("incompressible_model.jl")
include("show_incompressible_model.jl")
include("set_incompressible_model.jl")

#####
##### Time-stepping NonhydrostaticModels
#####

"""
    fields(model::NonhydrostaticModel)

Returns a flattened `NamedTuple` of the fields in `model.velocities` and `model.tracers`.
"""
fields(model::NonhydrostaticModel) = merge(model.velocities, model.tracers)
prognostic_fields(model::NonhydrostaticModel) = fields(model)

include("update_hydrostatic_pressure.jl")
include("update_incompressible_model_state.jl")
include("pressure_correction.jl")
include("velocity_and_tracer_tendencies.jl")
include("calculate_tendencies.jl")
include("correct_incompressible_immersed_tendencies.jl")

end # module
