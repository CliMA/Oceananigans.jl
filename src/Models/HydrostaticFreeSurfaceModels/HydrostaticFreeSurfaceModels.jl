module HydrostaticFreeSurfaceModels

export
    HydrostaticFreeSurfaceModel, VectorInvariant,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface, 
    PrescribedVelocityFields

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!

using DocStringExtensions

import Oceananigans: fields, prognostic_fields

# This is only used by the cubed sphere for now.
fill_horizontal_velocity_halos!(args...) = nothing

#####
##### HydrostaticFreeSurfaceModel definition
#####

FreeSurfaceDisplacementField(velocities, free_surface, grid) = Field{Center, Center, Nothing}(grid)
FreeSurfaceDisplacementField(velocities, ::Nothing, grid) = nothing

include("compute_w_from_continuity.jl")

include("rigid_lid.jl")

# Explicit free-surface solver functionality
include("explicit_free_surface.jl")

# Implicit free-surface solver functionality
include("compute_vertically_integrated_variables.jl")
include("pcg_implicit_free_surface_solver.jl")
include("matrix_implicit_free_surface_solver.jl")
include("fft_based_implicit_free_surface_solver.jl")
include("implicit_free_surface.jl")

# Split-Explicit free-surface solver functionality
include("split_explicit_free_surface.jl")
include("split_explicit_free_surface_kernels.jl")

include("hydrostatic_free_surface_field_tuples.jl")
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
@inline fields(model::HydrostaticFreeSurfaceModel) = merge(prognostic_fields(model), model.auxiliary_fields)

"""
    prognostic_fields(model::HydrostaticFreeSurfaceModel)

Returns a flattened `NamedTuple` of the prognostic fields associated with `HydrostaticFreeSurfaceModel`.
"""
@inline prognostic_fields(model::HydrostaticFreeSurfaceModel) =
    hydrostatic_prognostic_fields(model.velocities, model.free_surface, model.tracers)

@inline hydrostatic_prognostic_fields(velocities, free_surface, tracers) = merge((u = velocities.u,
                                                                                  v = velocities.v,
                                                                                  η = free_surface.η),
                                                                                  tracers)

@inline hydrostatic_prognostic_fields(velocities, ::Nothing, tracers) = merge((u = velocities.u,
                                                                               v = velocities.v),
                                                                               tracers)

displacement(free_surface) = free_surface.η
displacement(::Nothing) = nothing

include("barotropic_pressure_correction.jl")
include("hydrostatic_free_surface_advection.jl")
include("hydrostatic_free_surface_tendency_kernel_functions.jl")
include("calculate_hydrostatic_free_surface_tendencies.jl")
include("update_hydrostatic_free_surface_model_state.jl")
include("hydrostatic_free_surface_ab2_step.jl")
include("store_hydrostatic_free_surface_tendencies.jl")
include("prescribed_hydrostatic_velocity_fields.jl")
include("single_column_model_mode.jl")
include("slice_ensemble_model_mode.jl")

#####
##### Some diagnostics
#####

include("vertical_vorticity_field.jl")

end # module
