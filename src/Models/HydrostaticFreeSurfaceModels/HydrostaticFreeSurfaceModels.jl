module HydrostaticFreeSurfaceModels

export
    HydrostaticFreeSurfaceModel,
    ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface,
    PrescribedVelocityFields, ZStarCoordinate, ZCoordinate

using KernelAbstractions: @index, @kernel
using Adapt: Adapt

using Oceananigans.Architectures: architecture
using Oceananigans.Fields: ZFaceField
using Oceananigans.Grids: AbstractGrid, StaticVerticalDiscretization, OrthogonalSphericalShellGrid, Periodic, RectilinearGrid
using Oceananigans.Operators: Δzᶜᶠᶜ, Δzᶠᶜᶜ
using Oceananigans.TimeSteppers: TimeSteppers, SplitRungeKuttaTimeStepper, QuasiAdamsBashforth2TimeStepper
using Oceananigans.Utils: Utils, launch!, @apply_regionally

using DocStringExtensions: TYPEDFIELDS

import Oceananigans: fields, prognostic_fields, initialize!
import Oceananigans.Advection: cell_advection_timescale
import Oceananigans.Architectures: Architectures, on_architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Models: materialize_free_surface
import Oceananigans.Simulations: timestepper
import Oceananigans.TimeSteppers: step_lagrangian_particles!

# The only grid type that can support an FFT implicit free-surface solver
const XYRegularStaticRG = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:StaticVerticalDiscretization, <:Number, <:Number}

abstract type AbstractFreeSurface{E, G} end

struct ZCoordinate end
struct ZStarCoordinate end

Base.summary(::ZStarCoordinate) = "ZStarCoordinate"
Base.show(io::IO, c::ZStarCoordinate) = print(io, summary(c))

# This is only used by the cubed sphere for now.
fill_horizontal_velocity_halos!(args...) = nothing

#####
##### Utilities to compute the vertically integrated ``barotropic'' velocities
#####

# If U and V are prognostic (for example in `SplitExplicitFreeSurface`), we use them
@inline barotropic_U(i, j, k, grid, U, u) = @inbounds U[i, j, k]
@inline barotropic_V(i, j, k, grid, V, v) = @inbounds V[i, j, k]

# If either U or V are not available, we compute them
@inline function barotropic_U(i, j, k, grid, ::Nothing, u)
    U = zero(grid)
    for k in 1:size(grid, 3)
        @inbounds U += u[i, j, k] * Δzᶠᶜᶜ(i, j, k, grid)
    end
    return U
end

@inline function barotropic_V(i, j, k, grid, ::Nothing, v)
    V = zero(grid)
    for k in 1:size(grid, 3)
        @inbounds V += v[i, j, k] * Δzᶜᶠᶜ(i, j, k, grid)
    end
    return V
end

#####
##### HydrostaticFreeSurfaceModel definition
#####

free_surface_displacement_field(velocities, free_surface, grid) = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))
free_surface_displacement_field(velocities, ::Nothing, grid) = nothing

# free surface initialization functions
initialize_free_surface!(free_surface, grid, velocities) = nothing

# Transport velocity computation (only for a SplitExplicitFreeSurface)
compute_transport_velocities!(model, free_surface) = nothing

include("compute_w_from_continuity.jl")
include("hydrostatic_free_surface_field_tuples.jl")

# No free surface
include("nothing_free_surface.jl")

# Explicit free-surface solver functionality
include("explicit_free_surface.jl")

# Split-Explicit free-surface solver functionality
include("SplitExplicitFreeSurfaces/SplitExplicitFreeSurfaces.jl")
using .SplitExplicitFreeSurfaces

# Implicit free-surface solver functionality
include("fft_based_implicit_free_surface_solver.jl")
include("pcg_implicit_free_surface_solver.jl")
include("implicit_free_surface.jl")

# ZStarCoordinate implementation
include("z_star_vertical_spacing.jl")

# Hydrostatic model implementation
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

velocity_names(user_velocities) = (:u, :v, :w)

constructor_field_names(user_velocities, user_tracers, user_free_surface, auxiliary_fields, biogeochemistry, grid) =
    tuple(velocity_names(user_velocities)...,
          tracernames(user_tracers)...,
          free_surface_names(user_free_surface, user_velocities, grid)...,
          keys(auxiliary_fields)...,
          keys(biogeochemical_auxiliary_fields(biogeochemistry))...)

"""
    prognostic_fields(model::HydrostaticFreeSurfaceModel)

Return a flattened `NamedTuple` of the prognostic fields associated with `HydrostaticFreeSurfaceModel`.
"""
@inline prognostic_fields(model::HydrostaticFreeSurfaceModel) =
    hydrostatic_prognostic_fields(model.velocities, model.free_surface, model.tracers)

@inline horizontal_velocities(velocities) = (u=velocities.u, v=velocities.v)

# Note: we do not distinguish between prognostic and auxiliary free surface fields
# even though arguably the "filtered state" is an auxiliary part of the free surface state.
@inline free_surface_names(free_surface, velocities, grid) = tuple(:η)
@inline free_surface_names(free_surface::SplitExplicitFreeSurface, velocities, grid) = (:η, :U, :V)

@inline free_surface_fields(free_surface) = (; η=free_surface.displacement)
@inline free_surface_fields(::Nothing) = NamedTuple()
@inline free_surface_fields(free_surface::SplitExplicitFreeSurface) = (η = free_surface.displacement,
                                                                       U = free_surface.barotropic_velocities.U,
                                                                       V = free_surface.barotropic_velocities.V)

@inline hydrostatic_prognostic_fields(velocities, free_surface, tracers) =
    merge(horizontal_velocities(velocities), tracers, free_surface_fields(free_surface))

# Include vertical velocity
@inline hydrostatic_fields(velocities, free_surface, tracers) =
    merge((u=velocities.u, v=velocities.v, w=velocities.w),
          tracers,
          free_surface_fields(free_surface))

displacement(free_surface) = free_surface.displacement
displacement(::Nothing) = nothing

# Unpack model.particles to update particle properties. See Models/LagrangianParticleTracking/LagrangianParticleTracking.jl
TimeSteppers.step_lagrangian_particles!(model::HydrostaticFreeSurfaceModel, Δt) = step_lagrangian_particles!(model.particles, model, Δt)

include("barotropic_pressure_correction.jl")
include("hydrostatic_free_surface_tendency_kernel_functions.jl")
include("compute_hydrostatic_free_surface_tendencies.jl")
include("compute_hydrostatic_free_surface_buffers.jl")
include("compute_hydrostatic_flux_bcs.jl")
include("update_hydrostatic_free_surface_model_state.jl")
include("hydrostatic_free_surface_ab2_step.jl")
include("hydrostatic_free_surface_rk_step.jl")
include("cache_hydrostatic_free_surface_tendencies.jl")
include("prescribed_hydrostatic_velocity_fields.jl")
include("single_column_model_mode.jl")
include("slice_ensemble_model_mode.jl")

#####
##### Some diagnostics
#####

include("vertical_vorticity.jl")

end # module
