using KernelAbstractions: NoneEvent
using OffsetArrays: OffsetArray

using Oceananigans.Operators: Δzᵃᵃᶜ
using Oceananigans.BoundaryConditions: left_gradient, right_gradient, linearly_extrapolate, FBC, VBC, GBC
using Oceananigans.BoundaryConditions: fill_bottom_halo!, fill_top_halo!, apply_z_bottom_bc!, apply_z_top_bc!
using Oceananigans.Grids: Flat, Bounded
using Oceananigans.Architectures: device_event

import Oceananigans.Utils: launch!
import Oceananigans.Grids: validate_size, validate_halo
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.TurbulenceClosures: time_discretization
import Oceananigans.TurbulenceClosures: calculate_diffusivities!

#####
##### Implements a "single column model mode" for HydrostaticFreeSurfaceModel
#####

const SingleColumnGrid = AbstractGrid{<:AbstractFloat, <:Flat, <:Flat, <:Bounded}

#####
##### Model constructor utils
#####

PressureField(arch, ::SingleColumnGrid) = (pHY′ = nothing,)
FreeSurface(free_surface::ExplicitFreeSurface{Nothing}, velocities, arch, ::SingleColumnGrid) = nothing
FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, velocities, arch, ::SingleColumnGrid) = nothing

validate_momentum_advection(momentum_advection, ::SingleColumnGrid) = nothing
validate_tracer_advection(tracer_advection::AbstractAdvectionScheme, ::SingleColumnGrid) = nothing, NamedTuple()
validate_tracer_advection(tracer_advection::Nothing, ::SingleColumnGrid) = nothing, NamedTuple()

#####
##### Time-step optimizations
#####

calculate_free_surface_tendency!(arch, ::SingleColumnGrid, args...) = NoneEvent()

# Fast state update and halo filling

function update_state!(model::HydrostaticFreeSurfaceModel, grid::SingleColumnGrid)

    fill_halo_regions!(prognostic_fields(model), model.architecture, model.clock, fields(model))

    compute_auxiliary_fields!(model.auxiliary_fields)

    # Calculate diffusivities
    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)

    fill_halo_regions!(model.diffusivity_fields,
                       model.architecture,
                       model.clock,
                       fields(model))

    return nothing
end

import Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ, ∇_dot_qᶜ

using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure

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
validate_halo(TX, TY, TZ, e::ColumnEnsembleSize) = tuple(0, 0, e.Hz)

@inline time_discretization(::AbstractArray{<:AbstractTurbulenceClosure{TD}}) where TD = TD()

#####
##### TKEBasedVerticalDiffusivity helpers
#####

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVD, Kuᶜᶜᶜ, Kcᶜᶜᶜ, Keᶜᶜᶜ, _top_tke_flux, CATKEVDArray

import Oceananigans.TurbulenceClosures: calculate_diffusivities!
import Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: top_tke_flux

function calculate_diffusivities!(diffusivities, closure::CATKEVDArray, model)

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    clock = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    event = launch!(arch, grid, :xyz,
                    calculate_CATKEArray_diffusivities!,
                    diffusivities, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@kernel function calculate_CATKEArray_diffusivities!(diffusivities, grid::SingleColumnGrid, closure_array::CATKEVDArray, args...)
    i, j, k, = @index(Global, NTuple)
    @inbounds begin
        closure = closure_array[i, j]
        diffusivities.Kᵘ[i, j, k] = Kuᶜᶜᶜ(i, j, k, grid, closure, args...)
        diffusivities.Kᶜ[i, j, k] = Kcᶜᶜᶜ(i, j, k, grid, closure, args...)
        diffusivities.Kᵉ[i, j, k] = Keᶜᶜᶜ(i, j, k, grid, closure, args...)
    end
end

@inline tracer_tendency_kernel_function(model::HydrostaticFreeSurfaceModel, closure::CATKEVDArray, ::Val{:e}) =
    hydrostatic_turbulent_kinetic_energy_tendency

""" Compute the flux of TKE through the surface / top boundary. """
@inline function top_tke_flux(i, j, grid::SingleColumnGrid, clock, fields, parameters, closure_array::CATKEVDArray, buoyancy)
    top_tracer_bcs = parameters.top_tracer_boundary_conditions
    top_velocity_bcs = parameters.top_velocity_boundary_conditions
    @inbounds closure = closure_array[i, j]

    return _top_tke_flux(i, j, grid, closure.surface_TKE_flux, closure,
                         buoyancy, fields, top_tracer_bcs, top_velocity_bcs, clock)
end

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

import Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U

const FPlaneArray = AbstractArray{<:FPlane, 2}

@inline x_f_cross_U(i, j, k, grid::SingleColumnGrid, coriolis::FPlaneArray, U) = @inbounds - coriolis[i, j].f * ℑxyᶠᶜᵃ(i, j, k, grid, U[2])
@inline y_f_cross_U(i, j, k, grid::SingleColumnGrid, coriolis::FPlaneArray, U) = @inbounds   coriolis[i, j].f * ℑxyᶜᶠᵃ(i, j, k, grid, U[1])
@inline z_f_cross_U(i, j, k, grid::SingleColumnGrid, coriolis::FPlaneArray, U) = zero(eltype(grid))
