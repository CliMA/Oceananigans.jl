module IncompressibleModels

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!

import Oceananigans: fields, prognostic_fields

#####
##### IncompressibleModel definition
#####

include("incompressible_model.jl")
include("show_incompressible_model.jl")
include("set_incompressible_model.jl")

#####
##### Time-stepping IncompressibleModels
#####

"""
    fields(model::IncompressibleModel)

Returns a flattened `NamedTuple` of the fields in `model.velocities` and `model.tracers`.
"""
fields(model::IncompressibleModel) = merge(model.velocities, model.tracers)
prognostic_fields(model::IncompressibleModel) = fields(model)

include("update_hydrostatic_pressure.jl")
include("update_incompressible_model_state.jl")
include("pressure_correction.jl")
include("velocity_and_tracer_tendencies.jl")
include("calculate_tendencies.jl")
include("correct_incompressible_immersed_tendencies.jl")

end # module
