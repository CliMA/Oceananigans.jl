module HydrostaticFreeSurfaceModels

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!

import Oceananigans: fields

#####
##### HydrostaticFreeSurfaceModel definition
#####

include("hydrostatic_free_surface_tendency_fields.jl")
include("hydrostatic_free_surface_model.jl")
include("show_hydrostatic_free_surface_model.jl")
include("set_hydrostatic_free_surface_model.jl")

#####
##### Time-stepping HydrostaticFreeSurfaceModels
#####

"""
    fields(model::HydrostaticFreeSurfaceModel)

Returns a flattened `NamedTuple` of the fields in `model.velocities` and `model.tracers`.
"""
fields(model::HydrostaticFreeSurfaceModel) = merge((u = model.velocities.u,
                                                    v = model.velocities.v,
                                                    η = model.free_surface.η),
                                                    model.tracers)

include("update_hydrostatic_free_surface_model_state.jl")
include("hydrostatic_free_surface_time_step.jl")
#include("velocity_and_tracer_tendencies.jl")
#include("calculate_tendencies.jl")

end # module
