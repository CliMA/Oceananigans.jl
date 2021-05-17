using Oceananigans.Buoyancy: ∂z_b
using Oceananigans.Operators: ℑzᵃᵃᶜ

import Oceananigans: tracer_tendency_kernel_fwnction

"""
    struct RiDependentDiffusivityScales{FT}

A diffusivity model in which momentum, tracers, and TKE
each have Richardson-number-dependent diffusivities.
The Richardson number is

    ``Ri = ∂z B / ( (∂z U)² + (∂z V)² )`` ,

where ``B`` is buoyancy and ``∂z`` denotes a vertical derviative.
The Richardson-number dependent diffusivities are multiplied by the stability
function

    ``σ(Ri) = σ⁻ + (σ⁺ - σ⁻) * step(Ri, Riᶜ, Riʷ)``

where ``σ⁰``, ``σᵟ``, ``Riᶜ``, and ``Riʷ`` are free parameters,
and ``step`` is a smooth step function defined by

    ``step(x, c, w) = (1 + tanh((x - c) / w)) / 2``.
"""
Base.@kwdef struct RiDependentDiffusivityScales{FT}
    Cᴷu⁻  :: FT = 0.02
    Cᴷu⁺  :: FT = 0.01
    Cᴷc⁻  :: FT = 0.04
    Cᴷc⁺  :: FT = 0.01
    Cᴷe⁻  :: FT = 0.02
    Cᴷe⁺  :: FT = 0.01
    CᴷRiʷ :: FT = 0.1
    CᴷRiᶜ :: FT = 0.1
end

struct TKESurfaceFlux end

"""
    TKEBasedDiapycnalDiffusivity <: AbstractTurbulenceClosure{TD}

Parameters for the "anisotropic minimum dissipation" turbulence closure for large eddy simulation
proposed originally by [Rozema15](@cite) and [Abkar16](@cite), and then modified
by [Verstappen18](@cite), and finally described and validated for by [Vreugdenhil18](@cite).
"""
struct TKEBasedDiapycnalDiffusivity{TD, CK, CD, CL, CQ} <: AbstractTurbulenceClosure{TD}
        diffusivity_scales :: CK
      dissipation_constant :: CD
    mixing_length_constant :: CL
             surface_model :: CQ

    function TKEBasedDiapycnalDiffusivity{TD}(
            diffusivity_scales :: CK
          dissipation_constant :: CD
        mixing_length_constant :: CL
                 surface_model :: CQ) where {TD, CK, CD, CL, CQ}

        return new{TD, CK, CD, CL, CQ, D}(diffusivity_scales, dissipation_constant, mixing_length_constant, surface_model)
    end
end

function TKEBasedDiapycnalDiffusivity(FT=Float64;
                                      diffusivity_scales = RiDependentDiffusivityScales(),
                                      dissipation_constant = 2.0,
                                      mixing_length_constant = 0.7,
                                      surface_model = TKESurfaceFlux(),
                                      time_discretization::TD = ExplicitTimeDiscretization()) where TD

    return TKEBasedDiapycnalDiffusivity{TD}(diffusivity_scaling,
                                            dissipation_constant,
                                            mixing_length_constant,
                                            surface_tke_flux_model)
end

const TKEDD = TKEBasedDiapycnalDiffusivity

function turbulent_kinetic_energy_tendency end

# Hack right now...
tracer_tendency_kernel_function(model, closures::Tuple, ::Val{:e}) = tracer_tendency_kernel_function(model, closures[1], Val(:e))

tracer_tendency_kernel_function(model, closure::TKEBasedDiapycnalDiffusivity, ::Val{:e}) =
    hydrostatic_turbulent_kinetic_energy_tendency

# Note the superscript 'e'
@inline function ∇_dot_qᵉ(i, j, k, grid, clock, closure::TKEDD, e, tke_index, clock, velocities, tracers, buoyancy, diffusivities)
    κᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, κtkeᶜᶜᶜ, closure, tracers.e, velocities, tracers, buoyancy)
    return  κᶜᶜᶠ * ∂zᵃᵃᶠ(i, j, k, grid, e)
end

end

#####
##### Mixing length
#####

@inline depth(i, j, k, grid)               = grid.zᵃᵃᶠ(i, j, grid.Nz+1, grid) - grid.zᵃᵃᶜ(i, j, k, grid)
@inline height_above_bottom(i, j, k, grid) = grid.zᵃᵃᶜ(i, j, k, grid) - grid.zᵃᵃᶠ(i, j, 1, grid)

@inline wall_vertical_distance(i, j, k, grid) = min(depth(i, j, k, grid), height_above_bottom(i, j, k, grid)

@inline function buoyancy_mixing_length(i, j, k, grid, closure, e, tracers, buoyancy)
    FT = eltype(grid)
    Cᵇ = closure.mixing_length_constant
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    Ñ² = max(zero(FT), N²)
    return @inbounds ifelse(Ñ² <= 0, zero(FT), Cᵇ * sqrt(e[i, j, k] / Ñ²))
end

@inline function dissipation_mixing_length(i, j, k, grid, closure, e, tracers, buoyancy)
    ℓᶻ = wall_vertical_distance(i, j, k, grid)
    ℓᵇ = buoyancy_mixing_length(i, j, k, grid, closure, e, tracers, buoyancy)
    ℓ = min(ℓᶻ, ℓᵇ)
    ℓ_min = Δzᵃᵃᶜ(i, j, k, grid) / 2 # minimum mixing length...
    return max(δ, ℓ)
end

#####
##### Diffusivities
#####

@inline turbulent_velocity(i, j, k, grid, e) = @inbounds sqrt(max(zero(eltype(grid)), e[i, j, k]))

@inline function unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    ℓ = dissipation_mixing_length(i, j, k, grid, closure, e, tracers, buoyancy)
    u★ = turbulent_velocity(i, j, k, grid, e)
    return ℓ * u★
end

@inline function Kuᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    K = unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    σᵘ = momentum_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    return σᵘ * K
end

@inline function Kcᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    K = unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    σᶜ = tracer_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    return σᶜ * K
end

@inline function Keᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    K = unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    σᵉ = tke_diffusivity_scaling(i, j, k, grid, closure, velocities, tracers, buoyancy)
    return σᵉ * K
end

#####
##### Shear production
#####

@inline function u_shear_productionᶠᶜᶠ(i, j, k, grid, closure, clock, e, velocities, tracers, buoyancy, diffusivities)
    ∂z_u = ∂zᵃᵃᶠ(i, j, k, grid, velocities.u)
    w′u′ = viscous_flux_uz(i, j, k, grid, closure, clock, e, velocities, tracers, buoyancy, diffusivities)
    return - w′u′ * ∂z_u
end

@inline function v_shear_productionᶜᶠᶠ(i, j, k, grid, closure, clock, e, velocities, tracers, buoyancy, diffusivities)
    ∂z_v = ∂zᵃᵃᶠ(i, j, k, grid, velocities.v)
    w′v′ = viscous_flux_vz(i, j, k, grid, closure, clock, e, velocities, tracers, buoyancy, diffusivities)
    return - w′v′ * ∂z_v
end

# At ccc
@inline function shear_production(i, j, k, grid, closure, clock, e, velocities, tracers, buoyancy, diffusivities)
    return ℑxzᶜᵃᶜ(i, j, k, u_shear_production, closure, clock, e, velocities, tracers, buoyancy, diffusivities) +
           ℑyzᵃᶜᶜ(i, j, k, v_shear_production, closure, clock, e, velocities, tracers, buoyancy, diffusivities)
end

@inline function buoyancy_fluxᶜᶜᶠ(i, j, k, grid, clock, closure, e, velocities, tracers, buoyancy)
    κ = ℑzᵃᵃᶠ(i, j, k, grid, Kcᶜᶜᶜ, closure, e, velocities, tracers, buoyancy)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return - κ * N²
end

# at ccc
@inline buoyancy_flux(i, j, k, grid, closure, e, velocities, tracers, buoyancy) =
    ℑzᵃᵃᶜ(i, j, k, grid, buoyancy_fluxᶜᶜᶠ, closure, e, velocities, tracers, buoyancy)
    
@inline function dissipation(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    FT = eltype(grid)
    ℓ = dissipation_mixing_length(i, j, k, grid, closure, e, tracers, buoyancy)
    Cᴰ = closure.dissipation_constant
    @inbounds ẽ = max(zero(FT), e[i, j, k])
    three_halves = FT(3/2)
    return Cᴰ * ẽ^three_halves / ℓ
end

#####
##### Diffusive fluxes
#####

@inline function viscous_flux_uz(i, j, k, grid, closure::TKEDD, clock, velocities, tracers, buoyancy, diffusivities)
    νᶠᶜᶠ = ℑxzᶠᵃᶠ(i, j, k, grid, νᶜᶜᶜ, closure, tracers.e, velocities, tracers, buoyancy)
    return - νᶠᶜᶠ * ∂zᵃᵃᶠ(i, j, k, grid, velocities.u)
end

@inline function viscous_flux_vz(i, j, k, grid, closure::TKEDD, clock, velocities, tracers, buoyancy, diffusivities)
    νᶜᶠᶠ = ℑyzᵃᶠᶠ(i, j, k, grid, νᶜᶜᶜ, closure, tracers.e, velocities, tracers, buoyancy)
    return - νᶜᶠᶠ * ∂zᵃᵃᶠ(i, j, k, grid, velocities.v)
end

@inline function diffusive_flux_z(i, j, k, grid, closure::TKEDD, c, tracer_index, clock, velocities, tracers, buoyancy, diffusivities)
    κᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, κᶜᶜᶜ, closure, tracers.e, velocities, tracers, buoyancy)
    return  κᶜᶜᶠ * ∂zᵃᵃᶠ(i, j, k, grid, c)
end

