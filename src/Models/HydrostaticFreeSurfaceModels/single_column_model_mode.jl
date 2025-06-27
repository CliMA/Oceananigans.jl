using CUDA: @allowscalar

using Oceananigans: UpdateStateCallsite
using Oceananigans.Advection: AbstractAdvectionScheme
using Oceananigans.Grids: Flat, Bounded, ColumnEnsembleSize
using Oceananigans.Fields: ZeroField
using Oceananigans.Coriolis: AbstractRotation
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVDArray

import Oceananigans.Models: validate_tracer_advection
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.TurbulenceClosures: time_discretization, compute_diffusivities!
import Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∇_dot_qᶜ
import Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U

#####
##### Implements a "single column model mode" for HydrostaticFreeSurfaceModel
#####

const SingleColumnGrid = AbstractGrid{<:AbstractFloat, <:Flat, <:Flat, <:Bounded}

#####
##### Model constructor utils
#####

PressureField(arch, ::SingleColumnGrid) = (pHY′ = nothing,)
materialize_free_surface(::ExplicitFreeSurface{Nothing}, velocities,                 ::SingleColumnGrid) = nothing
materialize_free_surface(::ImplicitFreeSurface{Nothing}, velocities,                 ::SingleColumnGrid) = nothing
materialize_free_surface(::SplitExplicitFreeSurface,     velocities,                 ::SingleColumnGrid) = nothing
materialize_free_surface(::ExplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, ::SingleColumnGrid) = nothing
materialize_free_surface(::ImplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, ::SingleColumnGrid) = nothing
materialize_free_surface(::SplitExplicitFreeSurface,     ::PrescribedVelocityFields, ::SingleColumnGrid) = nothing

free_surface_names(::ExplicitFreeSurface{Nothing}, velocities,                 ::SingleColumnGrid) = tuple()
free_surface_names(::ImplicitFreeSurface{Nothing}, velocities,                 ::SingleColumnGrid) = tuple()
free_surface_names(::SplitExplicitFreeSurface,     velocities,                 ::SingleColumnGrid) = tuple()
free_surface_names(::ExplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, ::SingleColumnGrid) = tuple()
free_surface_names(::ImplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, ::SingleColumnGrid) = tuple()
free_surface_names(::SplitExplicitFreeSurface,     ::PrescribedVelocityFields, ::SingleColumnGrid) = tuple()

function hydrostatic_velocity_fields(::Nothing, grid::SingleColumnGrid, clock, bcs=NamedTuple())
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    w = ZeroField()
    return (u=u, v=v, w=w)
end

validate_velocity_boundary_conditions(::SingleColumnGrid, velocities) = nothing
validate_velocity_boundary_conditions(::SingleColumnGrid, ::PrescribedVelocityFields) = nothing
validate_momentum_advection(momentum_advection, ::SingleColumnGrid) = nothing
validate_tracer_advection(tracer_advection_tuple::NamedTuple, ::SingleColumnGrid) = Centered(), tracer_advection_tuple
validate_tracer_advection(tracer_advection::AbstractAdvectionScheme, ::SingleColumnGrid) = tracer_advection, NamedTuple()

compute_w_from_continuity!(velocities, arch, ::SingleColumnGrid; kwargs...) = nothing
compute_w_from_continuity!(::PrescribedVelocityFields, arch, ::SingleColumnGrid; kwargs...) = nothing

#####
##### Time-step optimizations
#####

# Disambiguation
compute_free_surface_tendency!(::SingleColumnGrid, model, ::ExplicitFreeSurface)      = nothing
compute_free_surface_tendency!(::SingleColumnGrid, model, ::SplitExplicitFreeSurface) = nothing

# Fast state update and halo filling

function update_state!(model::HydrostaticFreeSurfaceModel, grid::SingleColumnGrid, callbacks; compute_tendencies=true)

    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model))

    # Compute auxiliaries
    compute_auxiliary_fields!(model.auxiliary_fields)

    # Calculate diffusivities
    compute_diffusivities!(model.diffusivity_fields, model.closure, model)

    fill_halo_regions!(model.diffusivity_fields, model.clock, fields(model))

    for callback in callbacks
        callback.callsite isa UpdateStateCallsite && callback(model)
    end

    update_biogeochemical_state!(model.biogeochemistry, model)

    compute_tendencies &&
        @apply_regionally compute_tendencies!(model, callbacks)

    return nothing
end

const ClosureArray = AbstractArray{<:AbstractTurbulenceClosure}

@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid::SingleColumnGrid, closure_array::ClosureArray, args...)
    @inbounds closure = closure_array[i, j]
    return ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, args...)
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid::SingleColumnGrid, closure_array::ClosureArray, args...)
    @inbounds closure = closure_array[i, j]
    return ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, args...)
end

@inline function ∇_dot_qᶜ(i, j, k, grid::SingleColumnGrid, closure_array::ClosureArray, c, tracer_index, args...)
    @inbounds closure = closure_array[i, j]
    return ∇_dot_qᶜ(i, j, k, grid, closure, c, tracer_index, args...)
end

@inline function time_discretization(closure_array::AbstractArray)
    first_closure = @allowscalar first(closure_array) # assumes all closures have same time-discretization
    return time_discretization(first_closure)
end

#####
##### CATKEVerticalDiffusivity helpers
#####

@inline tracer_tendency_kernel_function(model::HydrostaticFreeSurfaceModel, closure::CATKEVDArray, ::Val{:e}) =
    hydrostatic_turbulent_kinetic_energy_tendency

@inline function hydrostatic_turbulent_kinetic_energy_tendency(i, j, k, grid::SingleColumnGrid,
                                                               val_tracer_index::Val{tracer_index},
                                                               advection,
                                                               closure_array::CATKEVDArray, args...) where tracer_index

    @inbounds closure = closure_array[i, j]
    return hydrostatic_turbulent_kinetic_energy_tendency(i, j, k, grid, val_tracer_index, advection, closure, args...)
end

#####
##### Arrays of Coriolises
#####

const CoriolisArray = AbstractArray{<:AbstractRotation}

@inline function x_f_cross_U(i, j, k, grid::SingleColumnGrid, coriolis_array::CoriolisArray, U)
    @inbounds coriolis = coriolis_array[i, j]
    return x_f_cross_U(i, j, k, grid, coriolis, U)
end

@inline function y_f_cross_U(i, j, k, grid::SingleColumnGrid, coriolis_array::CoriolisArray, U)
    @inbounds coriolis = coriolis_array[i, j]
    return y_f_cross_U(i, j, k, grid, coriolis, U)
end
