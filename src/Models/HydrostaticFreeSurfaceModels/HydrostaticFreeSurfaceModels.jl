module HydrostaticFreeSurfaceModels

export HydrostaticFreeSurfaceModel, VectorInvariant, ExplicitFreeSurface, ImplicitFreeSurface

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!

import Oceananigans: fields

#####
##### HydrostaticFreeSurfaceModel definition
#####

FreeSurfaceDisplacementField(velocities, arch, grid) = ReducedField(Center, Center, Nothing, arch, grid; dims=3)

include("compute_w_from_continuity.jl")
include("explicit_free_surface.jl")
include("implicit_free_surface.jl")
include("implicit_free_surface_solver.jl")
include("implicit_free_surface_solver_kernels.jl")
include("rigid_lid.jl")
include("hydrostatic_free_surface_velocity_fields.jl")
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
fields(model::HydrostaticFreeSurfaceModel) = hydrostatic_prognostic_fields(model.velocities, model.free_surface, model.tracers)

"""
    fields(model::HydrostaticFreeSurfaceModel)

Returns a flattened `NamedTuple` of the fields in `model.velocities` and `model.tracers`.
"""
hydrostatic_prognostic_fields(velocities, free_surface, tracers) = merge((u = velocities.u,
                                                                          v = velocities.v,
                                                                          η = free_surface.η),
                                                                          tracers)

displacement(free_surface) = free_surface.η
displacement(::Nothing) = nothing

include("barotropic_pressure_correction.jl")
include("hydrostatic_free_surface_advection.jl")
include("hydrostatic_free_surface_tendency_kernel_functions.jl")
include("calculate_hydrostatic_free_surface_tendencies.jl")
include("update_hydrostatic_free_surface_model_state.jl")
include("hydrostatic_free_surface_ab2_step.jl")
include("prescribed_hydrostatic_velocity_fields.jl")

#####
##### Some diagnostics
#####

include("vertical_vorticity_field.jl")

end # module
