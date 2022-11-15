using Oceananigans.Operators
using ..TurbulenceClosures: wall_vertical_distanceᶜᶜᶠ

"""
    struct MixingLength{FT}

Contains mixing length parameters for CATKE vertical diffusivity.

The mixing length is the sum of two mixing lengths:

```math
ℓᵩ = ℓʰ + ℓ⋆
```            

where ``ℓʰ`` is a convective mixing length,
and ``ℓ⋆`` is a mixing length in stably-stratified mixing.

Convective mixing length
========================

```math
ℓʰ = convecting ? Cᴬ * Δz : 0
```

Stably-stratified mixing length
===============================

```math
ℓ⋆ = σ * min(d, Cᵇ * √e / N, Cˢ * √e / S)
```

where ``S² = (∂z u)^2 + (∂z v)^2`` is the shear
frequency.

σ(Ri)`` is a stability function that depends on
the local Richardson number,

    ``Ri = ∂z B / S²``,

for buoyancy ``B``.

The Richardson-number dependent diffusivities are multiplied by the stability
function

    1. ``σ(Ri) = σ⁻ + (σ⁺ - σ⁻) * step(Ri, Riᶜ, Riʷ))``

where ``σ⁻``, ``σ⁺``, ``Riᶜ``, and ``Riʷ`` are free parameters,
and ``step`` is the piecewise linear function

    ``step(x, c, w) = max(0, min(1, (x - c) / w))``

"""
Base.@kwdef struct MixingLength{FT}
    Cᵇu  :: FT = Inf
    Cᵇc  :: FT = Inf
    Cᵇe  :: FT = Inf
    Cˢu  :: FT = Inf
    Cˢc  :: FT = Inf
    Cˢe  :: FT = Inf
    Cᴬu  :: FT = 0.0
    Cᴬc  :: FT = 0.0
    Cᴬe  :: FT = 0.0
    Cᴬˢ  :: FT = 0.0
    Cᴷu⁻ :: FT = 1.0
    Cᴷu⁺ :: FT = 1.0
    Cᴷc⁻ :: FT = 1.0
    Cᴷc⁺ :: FT = 1.0
    Cᴷe⁻ :: FT = 1.0
    Cᴷe⁺ :: FT = 1.0
    CRiʷ :: FT = 1.0
    CRiᶜ :: FT = 0.0
end

#####
##### Mixing length
#####

@inline ϕ⁺(i, j, k, grid, ψ) = @inbounds clip(ψ[i, j, k])
@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function buoyancy_mixing_lengthᶜᶜᶠ(i, j, k, grid, e, tracers, buoyancy)
    FT = eltype(grid)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = clip(N²)
    e⁺ = ℑzᵃᵃᶠ(i, j, k, grid, ϕ⁺, e)
    return ifelse(N²⁺ == 0, FT(Inf), sqrt(e⁺ / N²⁺))
end

@inline function shear_mixing_lengthᶜᶜᶠ(i, j, k, grid, e, velocities, tracers, buoyancy)
    FT = eltype(grid)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    S² = ∂z_u² + ∂z_v²
    e⁺ = ℑzᵃᵃᶠ(i, j, k, grid, ϕ⁺, e)
    return ifelse(S² == 0, FT(Inf), sqrt(e⁺ / S²))
end

@inline function stable_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᵇ::Number, Cˢ::Number, e, velocities, tracers, buoyancy)
    d = wall_vertical_distanceᶜᶜᶠ(i, j, k, grid)
    ℓᵇ = Cᵇ * buoyancy_mixing_lengthᶜᶜᶠ(i, j, k, grid, e, tracers, buoyancy)
    ℓˢ = Cˢ * shear_mixing_lengthᶜᶜᶠ(i, j, k, grid, e, velocities, tracers, buoyancy)
    return min(d, ℓᵇ, ℓˢ)
end

@inline function convective_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᴬ::Number, Cᴬˢ::Number,
                                             velocities, tracers, buoyancy, clock, tracer_bcs)

    # A kind of convective adjustment...
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers) # buoyancy frequency
    N²_above = ∂z_b(i, j, k+1, grid, buoyancy, tracers) # buoyancy frequency
    convecting = (N² < 0) | (N²_above < 0)
    ℓʰ = Cᴬ * Δzᶜᶜᶠ(i, j, k, grid)

    # "Sheared convection number"
    Qᵇ = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, merge(velocities, tracers))
    e⁺ = ℑzᵃᵃᶠ(i, j, k, grid, ϕ⁺, tracers.e)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    S = sqrt(∂z_u² + ∂z_v²)
    α = ifelse(Qᵇ > 0, S * Qᵇ / e⁺, zero(S))

    # "Shear aware" mising length
    ℓʰ *= 1 - Cᴬˢ * α
    ℓʰ = max(zero(grid), ℓʰ)
    
    return ifelse(convecting, ℓʰ, zero(grid))
end

"""Piecewise linear function between 0 (when x < c) and 1 (when x - c > w)."""
@inline step(x, c, w) = max(zero(x), min(one(x), (x - c) / w))
@inline scale(Ri, σ⁻, σ⁺, c, w)    = σ⁻ + (σ⁺ - σ⁻) * step(Ri, c, w)
@inline scale(Ri, σ⁻, σ⁺, c, w, n) = σ⁻ + (σ⁺ - σ⁻) * step(Ri, c, w)^n

@inline function stable_mixing_scale(i, j, k, grid, Cᴷ⁻, Cᴷ⁺, closure, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    CRiᶜ = closure.mixing_length.CRiᶜ
    CRiʷ = closure.mixing_length.CRiʷ
    return scale(Ri, Cᴷ⁻, Cᴷ⁺, CRiᶜ, CRiʷ)
end

@inline function momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    Cᴬ = closure.mixing_length.Cᴬu
    Cᴬˢ = closure.mixing_length.Cᴬˢ
    ℓʰ = convective_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᴬ, Cᴬˢ, velocities, tracers, buoyancy, clock, tracer_bcs)

    Cᴷ⁻ = closure.mixing_length.Cᴷu⁻
    Cᴷ⁺ = closure.mixing_length.Cᴷu⁺
    σ = stable_mixing_scale(i, j, k, grid, Cᴷ⁻, Cᴷ⁺, closure, velocities, tracers, buoyancy)

    Cᵇ = closure.mixing_length.Cᵇu
    Cˢ = closure.mixing_length.Cˢu
    ℓ★ = σ * stable_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᵇ, Cˢ, tracers.e, velocities, tracers, buoyancy)

    return ℓ★ + ℓʰ
end

@inline function tracer_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    Cᴬ = closure.mixing_length.Cᴬc
    Cᴬˢ = closure.mixing_length.Cᴬˢ
    ℓʰ = convective_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᴬ, Cᴬˢ, velocities, tracers, buoyancy, clock, tracer_bcs)

    Cᴷ⁻ = closure.mixing_length.Cᴷc⁻
    Cᴷ⁺ = closure.mixing_length.Cᴷc⁺
    σ = stable_mixing_scale(i, j, k, grid, Cᴷ⁻, Cᴷ⁺, closure, velocities, tracers, buoyancy)

    Cᵇ = closure.mixing_length.Cᵇc
    Cˢ = closure.mixing_length.Cˢc
    ℓ★ = σ * stable_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᵇ, Cˢ, tracers.e, velocities, tracers, buoyancy)

    return ℓ★ + ℓʰ
end

@inline function TKE_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    Cᴬ = closure.mixing_length.Cᴬe
    Cᴬˢ = closure.mixing_length.Cᴬˢ
    ℓʰ = convective_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᴬ, Cᴬˢ, velocities, tracers, buoyancy, clock, tracer_bcs)

    Cᴷ⁻ = closure.mixing_length.Cᴷe⁻
    Cᴷ⁺ = closure.mixing_length.Cᴷe⁺
    σ = stable_mixing_scale(i, j, k, grid, Cᴷ⁻, Cᴷ⁺, closure, velocities, tracers, buoyancy)

    Cᵇ = closure.mixing_length.Cᵇe
    Cˢ = closure.mixing_length.Cˢe
    ℓ★ = σ * stable_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᵇ, Cˢ, tracers.e, velocities, tracers, buoyancy)

    return ℓ★ + ℓʰ
end

Base.show(io::IO, ML::MixingLength) =
    print(io, "CATKEVerticalDiffusivities.MixingLength parameters:", "\n",
              "     Cᵇu  = $(ML.Cᵇu)",   "\n",
              "     Cᵇc  = $(ML.Cᵇc)",   "\n",
              "     Cᵇe  = $(ML.Cᵇe)",   "\n",
              "     Cˢu  = $(ML.Cˢu)",   "\n",
              "     Cˢc  = $(ML.Cˢc)",   "\n",
              "     Cˢe  = $(ML.Cˢe)",   "\n",
              "     Cᴬu  = $(ML.Cᴬu)",   "\n",
              "     Cᴬc  = $(ML.Cᴬc)",   "\n",
              "     Cᴬe  = $(ML.Cᴬe)",   "\n",
              "     Cᴷu⁻ = $(ML.Cᴷu⁻)",  "\n",
              "     Cᴷc⁻ = $(ML.Cᴷc⁻)",  "\n",
              "     Cᴷe⁻ = $(ML.Cᴷe⁻)",  "\n",
              "     Cᴷu⁺ = $(ML.Cᴷu⁺)",  "\n",
              "     Cᴷc⁺ = $(ML.Cᴷc⁺)",  "\n",
              "     Cᴷe⁺ = $(ML.Cᴷe⁺)",  "\n",
              "    CᴷRiʷ = $(ML.CᴷRiʷ)", "\n",
              "    CᴷRiᶜ = $(ML.CᴷRiᶜ)")

