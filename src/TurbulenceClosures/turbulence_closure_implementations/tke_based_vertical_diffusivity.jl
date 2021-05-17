using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators: ℑzᵃᵃᶜ

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
    TKEBasedVerticalDiffusivity <: AbstractTurbulenceClosure{TD}

Parameters for the "anisotropic minimum dissipation" turbulence closure for large eddy simulation
proposed originally by [Rozema15](@cite) and [Abkar16](@cite), and then modified
by [Verstappen18](@cite), and finally described and validated for by [Vreugdenhil18](@cite).
"""
struct TKEBasedVerticalDiffusivity{TD, CK, CD, CL, CQ} <: AbstractTurbulenceClosure{TD}
        diffusivity_scales :: CK
      dissipation_constant :: CD
    mixing_length_constant :: CL
             surface_model :: CQ

    function TKEBasedVerticalDiffusivity{TD}(
            diffusivity_scales :: CK,
          dissipation_constant :: CD,
        mixing_length_constant :: CL,
                 surface_model :: CQ) where {TD, CK, CD, CL, CQ}

        return new{TD, CK, CD, CL, CQ}(diffusivity_scales, dissipation_constant, mixing_length_constant, surface_model)
    end
end

function TKEBasedVerticalDiffusivity(FT=Float64;
                                      diffusivity_scales = RiDependentDiffusivityScales(),
                                      dissipation_constant = 2.0,
                                      mixing_length_constant = 0.7,
                                      surface_model = TKESurfaceFlux(),
                                      time_discretization::TD = ExplicitTimeDiscretization()) where TD

    return TKEBasedVerticalDiffusivity{TD}(diffusivity_scales,
                                            dissipation_constant,
                                            mixing_length_constant,
                                            surface_model)
end

const TKEVD = TKEBasedVerticalDiffusivity

#####
##### Utilities
#####

with_tracers(tracers, closure::TKEVD) = closure

calculate_diffusivities!(K, arch, grid, closure::TKEVD, args...) = nothing

function hydrostatic_turbulent_kinetic_energy_tendency end


#####
##### Mixing length
#####

@inline bottom(i, j, k, grid)              = znode(Center(), Center(), Face(), i, j, 1, grid)
@inline depth(i, j, k, grid)               = znode(Center(), Center(), Face(), i, j, grid.Nz+1, grid) - znode(Center(), Center(), Center(), i, j, k, grid)
@inline height_above_bottom(i, j, k, grid) = znode(Center(), Center(), Center(), i, j, k, grid) - bottom(i, j, k, grid)

@inline wall_vertical_distance(i, j, k, grid) = min(depth(i, j, k, grid), height_above_bottom(i, j, k, grid))

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
    return max(ℓ_min, ℓ)
end

#####
##### Diffusivities
#####

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    FT = eltype(grid)
    ∂z_u² = ℑxzᶜᵃᶜ(i, j, k, grid, ϕ², ∂zᵃᵃᶠ, velocities.u)
    ∂z_v² = ℑyzᵃᶜᶜ(i, j, k, grid, ϕ², ∂zᵃᵃᶠ, velocities.v)
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    return ifelse(N² == 0, zero(FT), N² / (∂z_u² + ∂z_v²))
end

@inline step(x, c, w) = (1 + tanh((x - c) / w)) / 2

@inline scale(Ri, σ⁻, σ⁺, c, w) = σ⁻ + (σ⁺ - σ⁻) * step(Ri, c, w)

@inline function momentum_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    return scale(Ri,
                 closure.diffusivity_scales.Cᴷu⁻,
                 closure.diffusivity_scales.Cᴷu⁺,
                 closure.diffusivity_scales.CᴷRiᶜ,
                 closure.diffusivity_scales.CᴷRiʷ)
end

@inline function tracer_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    return scale(Ri,
                 closure.diffusivity_scales.Cᴷc⁻,
                 closure.diffusivity_scales.Cᴷc⁺,
                 closure.diffusivity_scales.CᴷRiᶜ,
                 closure.diffusivity_scales.CᴷRiʷ)
end

@inline function TKE_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    return scale(Ri,
                 closure.diffusivity_scales.Cᴷe⁻,
                 closure.diffusivity_scales.Cᴷe⁺,
                 closure.diffusivity_scales.CᴷRiᶜ,
                 closure.diffusivity_scales.CᴷRiʷ)
end

@inline turbulent_velocity(i, j, k, grid, e) = @inbounds sqrt(max(zero(eltype(grid)), e[i, j, k]))

@inline function unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    ℓ = dissipation_mixing_length(i, j, k, grid, closure, e, tracers, buoyancy)
    u★ = turbulent_velocity(i, j, k, grid, e)
    return ℓ * u★
end

@inline function Kuᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    K = unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    σu = momentum_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    return σu * K
end

@inline function Kcᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    K = unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    σc = tracer_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    return σc * K
end

@inline function Keᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    K = unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    σe = TKE_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    return σe * K
end

#####
##### Shear production
#####

@inline function u_shear_productionᶠᶜᶠ(i, j, k, grid, closure, clock, velocities, tracers, buoyancy, diffusivities)
    ∂z_u = ∂zᵃᵃᶠ(i, j, k, grid, velocities.u)
    w′u′ = viscous_flux_uz(i, j, k, grid, closure, clock, velocities, diffusivities, tracers, buoyancy)
    return - w′u′ * ∂z_u
end

@inline function v_shear_productionᶜᶠᶠ(i, j, k, grid, closure, clock, velocities, tracers, buoyancy, diffusivities)
    ∂z_v = ∂zᵃᵃᶠ(i, j, k, grid, velocities.v)
    w′v′ = viscous_flux_vz(i, j, k, grid, closure, clock, velocities, diffusivities, tracers, buoyancy)
    return - w′v′ * ∂z_v
end

# At ccc
@inline function shear_production(i, j, k, grid, closure, clock, velocities, tracers, buoyancy, diffusivities)
    return ℑxzᶜᵃᶜ(i, j, k, grid, u_shear_productionᶠᶜᶠ, closure, clock, velocities, tracers, buoyancy, diffusivities) +
           ℑyzᵃᶜᶜ(i, j, k, grid, v_shear_productionᶜᶠᶠ, closure, clock, velocities, tracers, buoyancy, diffusivities)
end

@inline function buoyancy_fluxᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy)
    Kc = ℑzᵃᵃᶠ(i, j, k, grid, Kcᶜᶜᶜ, closure, tracers.e, velocities, tracers, buoyancy)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return - Kc * N²
end

# at ccc
@inline buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy) =
    ℑzᵃᵃᶜ(i, j, k, grid, buoyancy_fluxᶜᶜᶠ, closure, velocities, tracers, buoyancy)
    
@inline function dissipation(i, j, k, grid, closure, tracers, buoyancy)
    e = tracers.e
    FT = eltype(grid)
    ℓ = dissipation_mixing_length(i, j, k, grid, closure, e, tracers, buoyancy)
    Cᴰ = closure.dissipation_constant
    @inbounds ẽ = max(zero(FT), e[i, j, k])
    three_halves = FT(3/2)
    return Cᴰ * ẽ^three_halves / ℓ
end

#####
##### Viscous flux, diffusive fluxes, plus shenanigans for diffusive fluxes of TKE (eg TKE "transport")
#####

# Special "index type" alternative to Val for dispatch
struct TKETracerIndex{N} end
@inline TKETracerIndex(N) = TKETracerIndex{N}()

@inline function viscous_flux_uz(i, j, k, grid, closure::TKEVD, clock, velocities, diffusivities, tracers, buoyancy)
    νᶠᶜᶠ = ℑxzᶠᵃᶠ(i, j, k, grid, Kuᶜᶜᶜ, closure, tracers.e, velocities, tracers, buoyancy)
    return - νᶠᶜᶠ * ∂zᵃᵃᶠ(i, j, k, grid, velocities.u)
end

@inline function viscous_flux_vz(i, j, k, grid, closure::TKEVD, clock, velocities, diffusivities, tracers, buoyancy)
    νᶜᶠᶠ = ℑyzᵃᶠᶠ(i, j, k, grid, Kuᶜᶜᶜ, closure, tracers.e, velocities, tracers, buoyancy)
    return - νᶜᶠᶠ * ∂zᵃᵃᶠ(i, j, k, grid, velocities.v)
end

@inline function diffusive_flux_z(i, j, k, grid, closure::TKEVD, c, tracer_index, clock, diffusivities, tracers, buoyancy, velocities)
    κᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, Kcᶜᶜᶜ, closure, tracers.e, velocities, tracers, buoyancy)
    return  κᶜᶜᶠ * ∂zᵃᵃᶠ(i, j, k, grid, c)
end

# Diffusive flux of TKE!
@inline function diffusive_flux_z(i, j, k, grid, closure::TKEVD, e, ::TKETracerIndex, clock, diffusivities, tracers, buoyancy, velocities)
    κᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, Keᶜᶜᶜ, closure, e, velocities, tracers, buoyancy)
    return κᶜᶜᶠ * ∂zᵃᵃᶠ(i, j, k, grid, e)
end

# "Translations" for diffusive transport by non-TKEVD closures
@inline diffusive_flux_x(i, j, k, grid, closure, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_x(i, j, k, grid, closure, e, Val(N), args...)
@inline diffusive_flux_y(i, j, k, grid, closure, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_y(i, j, k, grid, closure, e, Val(N), args...)
@inline diffusive_flux_z(i, j, k, grid, closure, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_z(i, j, k, grid, closure, e, Val(N), args...)

# Shortcuts --- TKEVD incurs no horizontal transport
@inline diffusive_flux_x(i, j, k, grid, ::TKEVD, args...) = zero(eltype(grid))
@inline diffusive_flux_y(i, j, k, grid, ::TKEVD, args...) = zero(eltype(grid))

