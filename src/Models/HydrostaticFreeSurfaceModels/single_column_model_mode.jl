using CUDA: @allowscalar

using Oceananigans: UpdateStateCallsite
using Oceananigans.Advection: AbstractAdvectionScheme
using Oceananigans.Grids: Flat, Bounded
using Oceananigans.Fields: ZeroField
using Oceananigans.Coriolis: AbstractRotation
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVDArray

import Oceananigans.Grids: validate_size, validate_halo
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
materialize_free_surface(free_surface::ExplicitFreeSurface{Nothing}, velocities,                 ::SingleColumnGrid) = nothing
materialize_free_surface(free_surface::ImplicitFreeSurface{Nothing}, velocities,                 ::SingleColumnGrid) = nothing
materialize_free_surface(free_surface::SplitExplicitFreeSurface,     velocities,                 ::SingleColumnGrid) = nothing
materialize_free_surface(free_surface::ExplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, ::SingleColumnGrid) = nothing
materialize_free_surface(free_surface::ImplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, ::SingleColumnGrid) = nothing
materialize_free_surface(free_surface::SplitExplicitFreeSurface,     ::PrescribedVelocityFields, ::SingleColumnGrid) = nothing

function HydrostaticFreeSurfaceVelocityFields(::Nothing, grid::SingleColumnGrid, clock, bcs=NamedTuple())
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    w = ZeroField()
    return (u=u, v=v, w=w)
end

validate_velocity_boundary_conditions(::SingleColumnGrid, velocities) = nothing
validate_velocity_boundary_conditions(::SingleColumnGrid, ::PrescribedVelocityFields) = nothing
validate_momentum_advection(momentum_advection, ::SingleColumnGrid) = nothing
validate_tracer_advection(tracer_advection_tuple::NamedTuple, ::SingleColumnGrid) = CenteredSecondOrder(), tracer_advection_tuple
validate_tracer_advection(tracer_advection::AbstractAdvectionScheme, ::SingleColumnGrid) = tracer_advection, NamedTuple()

compute_w_from_continuity!(velocities, arch, ::SingleColumnGrid; kwargs...) = nothing
compute_w_from_continuity!(::PrescribedVelocityFields, arch, ::SingleColumnGrid; kwargs...) = nothing

#####
##### Time-step optimizations
#####

compute_free_surface_tendency!(::SingleColumnGrid, args...) = nothing

# Disambiguation
compute_free_surface_tendency!(::SingleColumnGrid, ::ImplicitFreeSurfaceHFSM     , args...) = nothing
compute_free_surface_tendency!(::SingleColumnGrid, ::SplitExplicitFreeSurfaceHFSM, args...) = nothing

# Fast state update and halo filling

function update_state!(model::HydrostaticFreeSurfaceModel, grid::SingleColumnGrid, Δt, callbacks)

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
    
struct ColumnEnsembleSize{C<:Tuple{Int, Int}}
    ensemble :: C
    Nz :: Int
    Hz :: Int
end

ColumnEnsembleSize(; Nz, ensemble=(0, 0), Hz=1) = ColumnEnsembleSize(ensemble, Nz, Hz)

validate_size(TX, TY, TZ, e::ColumnEnsembleSize) = tuple(e.ensemble[1], e.ensemble[2], e.Nz)
validate_halo(TX, TY, TZ, size, e::ColumnEnsembleSize) = tuple(0, 0, e.Hz)

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
