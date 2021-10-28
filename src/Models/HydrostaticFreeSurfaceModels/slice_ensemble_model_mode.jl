using Oceananigans.Grids: Flat, Bounded
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: _top_tke_flux, CATKEVDArray

import Oceananigans.Grids: validate_size, validate_halo
import Oceananigans.TurbulenceClosures: time_discretization, calculate_diffusivities!, with_tracers
import Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ, ∇_dot_qᶜ
import Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: top_tke_flux
import Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U

#####
##### Implements a "single column model mode" for HydrostaticFreeSurfaceModel
#####

const YZSliceGrid = Union{AbstractGrid{<:AbstractFloat, <:Flat, <:Bounded, <:Bounded},
                          AbstractGrid{<:AbstractFloat, <:Flat, <:Periodic, <:Bounded}}

@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid::YZSliceGrid, closure_array::ClosureArray, args...)
    @inbounds closure = closure_array[i]
    return ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, args...)
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid::YZSliceGrid, closure_array::ClosureArray, args...)
    @inbounds closure = closure_array[i]
    return ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, args...)
end

@inline function ∇_dot_qᶜ(i, j, k, grid::YZSliceGrid, closure_array::ClosureArray, c, tracer_index, args...)
    @inbounds closure = closure_array[i]
    return ∇_dot_qᶜ(i, j, k, grid, closure, c, tracer_index, args...)
end
    
struct SliceEnsembleSize
    ensemble :: Int
    Ny :: Int
    Nz :: Int
    Hy :: Int
    Hz :: Int
end

SliceEnsembleSize(; size, ensemble=0, halo=(1, 1)) = SliceEnsembleSize(ensemble, size[1], size[2], halo[1], halo[2])

validate_size(TX, TY, TZ, e::SliceEnsembleSize) = tuple(e.ensemble[1], e.Ny, e.Nz)
validate_halo(TX, TY, TZ, e::SliceEnsembleSize) = tuple(0, e.Hy, e.Hz)

#####
##### CATKEVerticalDiffusivity helpers
#####

""" Compute the flux of TKE through the surface / top boundary. """
@inline function top_tke_flux(i, j, grid::YZSliceGrid, clock, fields, parameters, closure_array::CATKEVDArray, buoyancy)
    top_tracer_bcs = parameters.top_tracer_boundary_conditions
    top_velocity_bcs = parameters.top_velocity_boundary_conditions
    @inbounds closure = closure_array[i]

    return _top_tke_flux(i, j, grid, closure.surface_TKE_flux, closure,
                         buoyancy, fields, top_tracer_bcs, top_velocity_bcs, clock)
end

@inline function hydrostatic_turbulent_kinetic_energy_tendency(i, j, k, grid::YZSliceGrid,
                                                               val_tracer_index::Val{tracer_index},
                                                               advection,
                                                               closure_array::CATKEVDArray, args...) where tracer_index

    @inbounds closure = closure_array[i]
    return hydrostatic_turbulent_kinetic_energy_tendency(i, j, k, grid, val_tracer_index, advection, closure, args...)
end

#####
##### Arrays of Coriolises
#####

const CoriolisArray = AbstractArray{<:AbstractRotation}

@inline x_f_cross_U(i, j, k, grid::YZSliceGrid, coriolis::CoriolisArray, U) = @inbounds x_f_cross_U(i, j, k, grid, coriolis[i], U)
@inline y_f_cross_U(i, j, k, grid::YZSliceGrid, coriolis::CoriolisArray, U) = @inbounds y_f_cross_U(i, j, k, grid, coriolis[i], U)
@inline z_f_cross_U(i, j, k, grid::YZSliceGrid, coriolis::CoriolisArray, U) = @inbounds z_f_cross_U(i, j, k, grid, coriolis[i], U)

