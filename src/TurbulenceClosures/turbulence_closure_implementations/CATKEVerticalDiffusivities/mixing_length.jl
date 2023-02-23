using ..TurbulenceClosures: wall_vertical_distanceᶜᶜᶠ, total_depthᶜᶜᵃ

"""
    struct MixingLength{FT}

Contains mixing length parameters for CATKE vertical diffusivity.
"""
Base.@kwdef struct MixingLength{FT}
    Cᵇ   :: FT = Inf
    Cˢ   :: FT = Inf
    Cᶜc  :: FT = 0.0
    Cᶜe  :: FT = 0.0
    Cᵉc  :: FT = 0.0
    Cᵉe  :: FT = 0.0
    Cˢᶜ  :: FT = 0.0
    C⁻u  :: FT = 1.0
    C⁺u  :: FT = 1.0
    C⁻c  :: FT = 1.0
    C⁺c  :: FT = 1.0
    C⁻e  :: FT = 1.0
    C⁺e  :: FT = 1.0
    CRiʷ :: FT = 1.0
    CRiᶜ :: FT = 0.0
end

#####
##### Mixing length
#####

@inline ϕ⁺(i, j, k, grid, ψ) = @inbounds clip(ψ[i, j, k])
@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function shearᶜᶜᶠ(i, j, k, grid, velocities)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    S² = ∂z_u² + ∂z_v²
    return S²
end

@inline function buoyancy_mixing_lengthᶜᶜᶠ(i, j, k, grid, e, tracers, buoyancy)
    FT = eltype(grid)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = clip(N²)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocity, e)
    return ifelse(N²⁺ == 0, FT(Inf), w★ / sqrt(N²⁺))
end

@inline function shear_mixing_lengthᶜᶜᶠ(i, j, k, grid, e, velocities)
    FT = eltype(grid)
    S² = shearᶜᶜᶠ(i, j, k, grid, velocities)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocity, e)
    return ifelse(S² == 0, FT(Inf), w★ / sqrt(S²))
end

@inline function stable_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᵇ::Number, Cˢ::Number, e, velocities, tracers, buoyancy)
    ℓᵇ = Cᵇ * buoyancy_mixing_lengthᶜᶜᶠ(i, j, k, grid, e, tracers, buoyancy)
    d = wall_vertical_distanceᶜᶜᶠ(i, j, k, grid)
    ℓᵇ = ifelse(isnan(ℓᵇ), d, ℓᵇ)
    ℓ = min(d, ℓᵇ)
    return ℓ
end

@inline three_halves(i, j, k, grid, e) = @inbounds sqrt(clip(e[i, j, k])^3)

@inline function convective_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᶜ::Number, Cᵉ::Number, Cˢᶜ::Number,
                                             velocities, tracers, buoyancy, clock, tracer_bcs)

    Qᵇ  = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, merge(velocities, tracers))
    N²  = ∂z_b(i, j, k, grid, buoyancy, tracers)
    w★  = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocity, tracers.e)
    w★³ = ℑzᵃᵃᶠ(i, j, k, grid, three_halves, tracers.e)
    w★² = ℑzᵃᵃᶠ(i, j, k, grid, ϕ⁺, tracers.e)

    # "Convective length"
    # ℓᶜ ∼ boundary layer depth according to Deardorff scaling
    ℓᶜ = Cᶜ * w★³ / Qᵇ

    # "Entrainment length"
    # Ensures that w′b′ ~ Qᵇ at entrainment depth
    ℓᵉ = Cᵉ * Qᵇ / (w★ * N²)

    # Figure out which mixing length applies
    N²_above = ∂z_b(i, j, k+1, grid, buoyancy, tracers) # buoyancy frequency
    convecting = (Qᵇ > 0) & (N² < 0)
    entraining = (Qᵇ > 0) & (N² > 0) & (N²_above < 0)

    # Model for shear-convection interaction
    S² = shearᶜᶜᶠ(i, j, k, grid, velocities)
    Sc = sqrt(S²) * w★² / Qᵇ # Cs = "Sheared convection number"
    ϵᶜˢ = 1 - Cˢᶜ * Sc       # ϵ = Sheared convection factor
    
    # Reduce convective and entraining mixing lengths by sheared convection factor
    # end ensure non-negativity
    ℓᶜ = clip(ϵᶜˢ * ℓᶜ)
    ℓᵉ = clip(ϵᶜˢ * ℓᵉ)

    ℓ = ifelse(convecting, ℓᶜ,
        ifelse(entraining, ℓᵉ, zero(grid)))

    return ifelse(isnan(ℓ), zero(grid), ℓ)
end

"""Piecewise linear function between 0 (when x < c) and 1 (when x - c > w)."""
@inline step(x, c, w) = max(zero(x), min(one(x), (x - c) / w))
@inline scale(Ri, σ⁻, σ⁺, c, w) = σ⁻ + (σ⁺ - σ⁻) * step(Ri, c, w)

@inline function stable_mixing_scaleᶜᶜᶠ(i, j, k, grid, C⁻, C⁺, closure, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    CRiᶜ = closure.mixing_length.CRiᶜ
    CRiʷ = closure.mixing_length.CRiʷ
    return scale(Ri, C⁻, C⁺, CRiᶜ, CRiʷ)
end

@inline function momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    C⁻ = closure.mixing_length.C⁻u
    C⁺ = closure.mixing_length.C⁺u
    σ = stable_mixing_scaleᶜᶜᶠ(i, j, k, grid, C⁻, C⁺, closure, velocities, tracers, buoyancy)

    Cᵇ = closure.mixing_length.Cᵇ
    Cˢ = closure.mixing_length.Cˢ
    ℓ★ = σ * stable_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᵇ, Cˢ, tracers.e, velocities, tracers, buoyancy)

    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)

    H = total_depthᶜᶜᵃ(i, j, grid)
    return min(H, ℓ★)
end

@inline function tracer_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    Cᶜ  = closure.mixing_length.Cᶜc
    Cᵉ  = closure.mixing_length.Cᵉc
    Cˢᶜ = closure.mixing_length.Cˢᶜ
    ℓʰ = convective_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᶜ, Cᵉ, Cˢᶜ, velocities, tracers, buoyancy, clock, tracer_bcs)

    C⁻ = closure.mixing_length.C⁻c
    C⁺ = closure.mixing_length.C⁺c
    σ = stable_mixing_scaleᶜᶜᶠ(i, j, k, grid, C⁻, C⁺, closure, velocities, tracers, buoyancy)

    Cᵇ = closure.mixing_length.Cᵇ
    Cˢ = closure.mixing_length.Cˢ
    ℓ★ = σ * stable_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᵇ, Cˢ, tracers.e, velocities, tracers, buoyancy)

    ℓʰ = ifelse(isnan(ℓʰ), zero(grid), ℓʰ)
    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)

    H = total_depthᶜᶜᵃ(i, j, grid)
    return min(H, ℓ★ + ℓʰ)
end

@inline function TKE_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    Cᶜ  = closure.mixing_length.Cᶜe
    Cᵉ  = closure.mixing_length.Cᵉe
    Cˢᶜ = closure.mixing_length.Cˢᶜ
    ℓʰ = convective_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᶜ, Cᵉ, Cˢᶜ, velocities, tracers, buoyancy, clock, tracer_bcs)

    C⁻ = closure.mixing_length.C⁻e
    C⁺ = closure.mixing_length.C⁺e
    σ = stable_mixing_scaleᶜᶜᶠ(i, j, k, grid, C⁻, C⁺, closure, velocities, tracers, buoyancy)

    Cᵇ = closure.mixing_length.Cᵇ
    Cˢ = closure.mixing_length.Cˢ
    ℓ★ = σ * stable_mixing_lengthᶜᶜᶠ(i, j, k, grid, Cᵇ, Cˢ, tracers.e, velocities, tracers, buoyancy)

    ℓʰ = ifelse(isnan(ℓʰ), zero(grid), ℓʰ)
    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)

    H = total_depthᶜᶜᵃ(i, j, grid)
    return min(H, ℓ★ + ℓʰ)
end

Base.show(io::IO, ML::MixingLength) =
    print(io, "CATKEVerticalDiffusivities.MixingLength parameters:", '\n',
              "    Cᵇ   = $(ML.Cᵇ)",   '\n',
              "    Cˢ   = $(ML.Cˢ)",   '\n',
              "    Cᶜc  = $(ML.Cᶜc)",  '\n',
              "    Cᶜe  = $(ML.Cᶜe)",  '\n',
              "    Cᵉc  = $(ML.Cᵉc)",  '\n',
              "    Cᵉe  = $(ML.Cᵉe)",  '\n',
              "    C⁻u  = $(ML.C⁻u)", '\n',
              "    C⁻c  = $(ML.C⁻c)", '\n',
              "    C⁻e  = $(ML.C⁻e)", '\n',
              "    C⁺u  = $(ML.C⁺u)", '\n',
              "    C⁺c  = $(ML.C⁺c)", '\n',
              "    C⁺e  = $(ML.C⁺e)", '\n',
              "    CRiʷ = $(ML.CRiʷ)", '\n',
              "    CRiᶜ = $(ML.CRiᶜ)")

