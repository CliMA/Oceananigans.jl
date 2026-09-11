#####
##### ObliqueRadiation (Raymond & Kuo 1984) open boundary scheme
#####

"""
    ObliqueRadiation(; inflow_timescale = 0, outflow_timescale = Inf, use_boundary_velocity = false)

Raymond & Kuo (1984) two-dimensional radiation condition with adaptive nudging
(Marchesiello et al. 2001):

    ∂φ/∂t + cₙ ∂φ/∂n + cₜ ∂φ/∂t̂ = - (φ - φᵉˣᵗ) / τ

with

    cₙ = -(∂φ/∂t)(∂φ/∂n) / |∇φ|²,   cₜ = -(∂φ/∂t)(∂φ/∂t̂) / |∇φ|²,   |∇φ|² = (∂φ/∂n)² + (∂φ/∂t̂)²

where `n` is the boundary-normal and `t̂` the horizontal boundary-tangential direction. The
tangential derivative is upwinded and `cₜ` is limited to a tangential Courant number of one.
For `∂φ/∂t̂ = 0` the scheme reduces to `NormalRadiation`. The tangential term is applied on the
lateral boundaries only.

Inflow versus outflow is decided from the boundary-normal velocity: on inflow `cₙ = cₜ = 0` and
`τ = inflow_timescale`; on outflow `τ = outflow_timescale`. For `Value` boundary conditions,
`use_boundary_velocity = true` takes that velocity at the boundary face rather than one cell into
the interior.

References
==========
* Raymond, W. H., & Kuo, H. L. (1984). "A radiation boundary condition for multi-dimensional
  flows." Quarterly Journal of the Royal Meteorological Society, 110(464), 535-551.
* Marchesiello, P., McWilliams, J. C., & Shchepetkin, A. (2001). "Open boundary conditions
  for long-term integration of regional oceanic models." Ocean Modelling, 3(1-2), 1-20.

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: ObliqueRadiation

ObliqueRadiation()

# output
ObliqueRadiation{Float64}
├── inflow_timescale: 0.0
├── outflow_timescale: Inf
└── use_boundary_velocity: false
```
"""
struct ObliqueRadiation{FT, S} <: AbstractRadiationScheme{FT}
    outflow_timescale :: FT
    inflow_timescale  :: FT
    use_boundary_velocity :: Bool
    φᵇ  :: S
    φ₁  :: S
    φ₁ˡ :: S
end

function ObliqueRadiation(FT = defaults.FloatType;
                          inflow_timescale = 0,
                          outflow_timescale = Inf,
                          use_boundary_velocity = false)

    outflow_timescale = convert(FT, outflow_timescale)
    inflow_timescale = convert(FT, inflow_timescale)
    return ObliqueRadiation(outflow_timescale, inflow_timescale, use_boundary_velocity, nothing, nothing, nothing)
end

# Backward and forward differences along the boundary face, zero beyond its ends.
@inline function tangential_differences(φ, t, k)
    T = size(φ, 1)
    @inbounds begin
        φ₀ = φ[t, k]
        φ₋ = φ[max(t - 1, 1), k]
        φ₊ = φ[min(t + 1, T), k]
    end
    return φ₀ - φ₋, φ₊ - φ₀
end

@inline function raymond_kuo_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δᵇ₋, δᵇ₊, δ₁₋, δ₁₊, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
    ∂t_φ = φ₁ⁿ⁺¹ - φ₁ⁿ
    ∂ξ_φ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹
    ∂η_φ = ifelse(-∂t_φ * (δ₁₋ + δ₁₊) > 0, δ₁₋, δ₁₊)

    FT = typeof(∂ξ_φ)
    D  = max(∂ξ_φ^2 + ∂η_φ^2, eps(FT))
    Cᶜ = - ∂t_φ * ∂ξ_φ / D
    Cₙ = ifelse(outflow, max(zero(FT), min(one(FT), max(Cᶜ, Cᵃ))), zero(FT))
    Cₜ = ifelse(outflow, clamp(- ∂t_φ * ∂η_φ / D, -one(FT), one(FT)), zero(FT))

    τ = ifelse(outflow, radiation.outflow_timescale, radiation.inflow_timescale)
    τ̃ = Δt / τ

    φᵇⁿ⁺¹ = (φᵇⁿ + Cₙ * φ₁ⁿ⁺¹ - max(Cₜ, zero(FT)) * δᵇ₋ - min(Cₜ, zero(FT)) * δᵇ₊ + τ̃ * φᵉˣᵗ) / (1 + Cₙ + τ̃)

    return ifelse(τ == 0, φᵉˣᵗ, φᵇⁿ⁺¹)
end

@inline function radiation_update(radiation::ObliqueRadiation, t, k, φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, outflow, Cᵃ)
    δᵇ₋, δᵇ₊ = tangential_differences(radiation.φᵇ, t, k)
    δ₁₋, δ₁₊ = tangential_differences(radiation.φ₁, t, k)
    return raymond_kuo_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δᵇ₋, δᵇ₊, δ₁₋, δ₁₊, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
end
