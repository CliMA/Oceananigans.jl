module HydrostaticFreeSurfaceModels

export
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface, 
    PrescribedVelocityFields

using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils
using Oceananigans.Utils: launch!, SumOfArrays
using Oceananigans.Grids: AbstractGrid

using DocStringExtensions

import Oceananigans: fields, prognostic_fields, initialize!
import Oceananigans.Advection: cell_advection_timescale
import Oceananigans.TimeSteppers: step_lagrangian_particles!

abstract type AbstractFreeSurface{E, G} end

# This is only used by the cubed sphere for now.
fill_horizontal_velocity_halos!(args...) = nothing

#####
##### HydrostaticFreeSurfaceModel definition
#####

FreeSurfaceDisplacementField(velocities, free_surface, grid) = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))
FreeSurfaceDisplacementField(velocities, ::Nothing, grid) = nothing

include("compute_w_from_continuity.jl")
include("rigid_lid.jl")

# Explicit free-surface solver functionality
include("explicit_free_surface.jl")

# Implicit free-surface solver functionality
include("implicit_free_surface_utils.jl")
include("compute_vertically_integrated_variables.jl")
include("fft_based_implicit_free_surface_solver.jl")
include("pcg_implicit_free_surface_solver.jl")
include("matrix_implicit_free_surface_solver.jl")
include("implicit_free_surface.jl")

# Split-Explicit free-surface solver functionality
include("split_explicit_free_surface.jl")
include("distributed_split_explicit_free_surface.jl")
include("split_explicit_free_surface_kernels.jl")

include("hydrostatic_free_surface_field_tuples.jl")
include("hydrostatic_free_surface_model.jl")
include("show_hydrostatic_free_surface_model.jl")
include("set_hydrostatic_free_surface_model.jl")

#####
##### AbstractModel interface
#####

cell_advection_timescale(model::HydrostaticFreeSurfaceModel) = cell_advection_timescale(model.grid, model.velocities)

"""
    fields(model::HydrostaticFreeSurfaceModel)

Return a flattened `NamedTuple` of the fields in `model.velocities`, `model.free_surface`,
`model.tracers`, and any auxiliary fields for a `HydrostaticFreeSurfaceModel` model.
"""
@inline fields(model::HydrostaticFreeSurfaceModel) = 
    merge(hydrostatic_fields(model.velocities, model.free_surface, model.tracers),
          model.auxiliary_fields,
          biogeochemical_auxiliary_fields(model.biogeochemistry))

"""
    prognostic_fields(model::HydrostaticFreeSurfaceModel)

Return a flattened `NamedTuple` of the prognostic fields associated with `HydrostaticFreeSurfaceModel`.
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
                                               
@inline hydrostatic_fields(velocities, free_surface, tracers) = merge((u = velocities.u,
                                                                       v = velocities.v,
                                                                       w = velocities.w),
                                                                       tracers,
                                                                       (; η = free_surface.η))

@inline hydrostatic_fields(velocities, ::Nothing, tracers) = merge((u = velocities.u,
                                                                    v = velocities.v,
                                                                    w = velocities.w),
                                                                    tracers)

displacement(free_surface) = free_surface.η
displacement(::Nothing) = nothing

# Unpack model.particles to update particle properties. See Models/LagrangianParticleTracking/LagrangianParticleTracking.jl
step_lagrangian_particles!(model::HydrostaticFreeSurfaceModel, Δt) = step_lagrangian_particles!(model.particles, model, Δt)

include("barotropic_pressure_correction.jl")
include("hydrostatic_free_surface_tendency_kernel_functions.jl")
include("compute_hydrostatic_free_surface_tendencies.jl")
include("compute_hydrostatic_free_surface_boundary_tendencies.jl")
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
