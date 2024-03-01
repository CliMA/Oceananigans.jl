using ..TurbulenceClosures:
    wall_vertical_distanceᶜᶜᶠ,
    wall_vertical_distanceᶜᶜᶜ,
    depthᶜᶜᶠ,
    height_above_bottomᶜᶜᶠ,
    depthᶜᶜᶜ,
    height_above_bottomᶜᶜᶜ,
    total_depthᶜᶜᵃ

"""
    struct MixingLength{FT}

Contains mixing length parameters for CATKE vertical diffusivity.
"""
Base.@kwdef struct MixingLength{FT}
    Cˢ   :: FT = 1.03   # Surface distance coefficient for shear length scale
    Cᵇ   :: FT = Inf    # Bottom distance coefficient for shear length scale
    Cᶜc  :: FT = 3.72   # Convective mixing length coefficient for tracers
    Cᶜe  :: FT = 1.40   # Convective mixing length coefficient for TKE
    Cᵉc  :: FT = 0.841  # Convective penetration mixing length coefficient for tracers
    Cᵉe  :: FT = 0.0    # Convective penetration mixing length coefficient for TKE
    Cˢᵖ  :: FT = 1.02   # Sheared convective plume coefficient
    Cˡᵒu :: FT = 0.775  # Shear mixing length coefficient for momentum at low Ri
    Cʰⁱu :: FT = 0.765  # Shear mixing length coefficient for momentum at high Ri
    Cˡᵒc :: FT = 0.786  # Shear mixing length coefficient for tracers at low Ri
    Cʰⁱc :: FT = 0.378  # Shear mixing length coefficient for tracers at high Ri
    Cˡᵒe :: FT = 2.68   # Shear mixing length coefficient for TKE at low Ri
    Cʰⁱe :: FT = 2.16   # Shear mixing length coefficient for TKE at high Ri
    CRiᵟ :: FT = 0.421  # Stability function width 
    CRi⁰ :: FT = 0.261  # Stability function lower Ri
end

#####
##### Mixing length
#####

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function shearᶜᶜᶠ(i, j, k, grid, u, v)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, v)
    S² = ∂z_u² + ∂z_v²
    return S²
end

@inline function shearᶜᶜᶜ(i, j, k, grid, u, v)
    ∂z_u² = ℑxzᶜᵃᶜ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, u)
    ∂z_v² = ℑyzᵃᶜᶜ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, v)
    S² = ∂z_u² + ∂z_v²
    return S²
end

@inline function stratification_mixing_lengthᶜᶜᶠ(i, j, k, grid, N², w★)
    FT = eltype(grid)
    @inbounds N²⁺ = clip(N²[i, j, k])
    w★ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, w★)
    return ifelse(N²⁺ == 0, FT(Inf), w★ᶜᶜᶠ / sqrt(N²⁺))
end

@inline function stratification_mixing_lengthᶜᶜᶜ(i, j, k, grid, N², w★)
    FT = eltype(grid)
    N²ᶜᶜᶜ = ℑzᵃᵃᶜ(i, j, k, grid, N²)
    N²⁺ = clip(N²ᶜᶜᶜ)
    w★ᶜᶜᶜ = @inbounds w★[i, j, k]
    return ifelse(N²⁺ == 0, FT(Inf), w★ᶜᶜᶜ / sqrt(N²⁺))
end

@inline function stable_length_scaleᶜᶜᶠ(i, j, k, grid, closure, N², w★)
    Cˢ = closure.mixing_length.Cˢ
    Cᵇ = closure.mixing_length.Cᵇ

    d_up   = Cˢ * depthᶜᶜᶠ(i, j, k, grid)
    d_down = Cᵇ * height_above_bottomᶜᶜᶠ(i, j, k, grid)
    d = min(d_up, d_down)

    ℓᴺ = stratification_mixing_lengthᶜᶜᶠ(i, j, k, grid, N², w★)

    ℓ = min(d, ℓᴺ)
    ℓ = ifelse(isnan(ℓ), d, ℓ)

    return ℓ
end

@inline function stable_length_scaleᶜᶜᶜ(i, j, k, grid, closure, N², w★)
    Cˢ = closure.mixing_length.Cˢ
    Cᵇ = closure.mixing_length.Cᵇ
    d_up   = Cˢ * depthᶜᶜᶜ(i, j, k, grid)
    d_down = Cᵇ * height_above_bottomᶜᶜᶜ(i, j, k, grid)
    d = min(d_up, d_down)

    ℓᴺ = stratification_mixing_lengthᶜᶜᶜ(i, j, k, grid, N², w★)

    ℓ = min(d, ℓᴺ)
    ℓ = ifelse(isnan(ℓ), d, ℓ)

    return ℓ
end

@inline three_halves_tkeᶜᶜᶜ(i, j, k, grid, w★) = @inbounds w★[i, j, k]^3
@inline squared_tkeᶜᶜᶜ(i, j, k, grid, w★) = @inbounds w★[i, j, k]^2

@inline function convective_length_scaleᶜᶜᶠ(i, j, k, grid, closure, Cᶜ::Number, Cᵉ::Number, Cˢᵖ::Number,
                                            surface_buoyancy_flux, S², N², w★)

    Qᵇᵋ      = closure.minimum_convective_buoyancy_flux
    Qᵇ       = @inbounds surface_buoyancy_flux[i, j, 1]
    w★¹ᶜᶜᶠ    = ℑzᵃᵃᶠ(i, j, k, grid, w★)
    w★²ᶜᶜᶠ   = ℑzᵃᵃᶠ(i, j, k, grid, squared_tkeᶜᶜᶜ, w★)
    w★³ᶜᶜᶠ   = ℑzᵃᵃᶠ(i, j, k, grid, three_halves_tkeᶜᶜᶜ, w★)
    S²ᶜᶜᶠ    = @inbounds S²[i, j, k]
    N²_local = @inbounds N²[i, j, k]
    N²_above = @inbounds N²[i, j, k+1]

    # "Convective length"
    # ℓᶜ ∼ boundary layer depth according to Deardorff scaling
    ℓᶜ = Cᶜ * w★³ᶜᶜᶠ / (Qᵇ + Qᵇᵋ)
    ℓᶜ = ifelse(isnan(ℓᶜ), zero(grid), ℓᶜ)

    # Figure out which mixing length applies
    convecting = (Qᵇ > Qᵇᵋ) & (N²_local < 0)

    # Model for shear-convection interaction
    Sp = sqrt(S²ᶜᶜᶠ) * w★²ᶜᶜᶠ / (Qᵇ + Qᵇᵋ) # Sp = "Sheared convection number"
    ϵˢᵖ = 1 - Cˢᵖ * Sp               # ϵ = Sheared convection factor
    
    # Reduce convective and entraining mixing lengths by sheared convection factor
    # end ensure non-negativity
    ℓᶜ = clip(ϵˢᵖ * ℓᶜ)

    # "Entrainment length"
    # Ensures that w′b′ ~ Qᵇ at entrainment depth
    ℓᵉ = Cᵉ * Qᵇ / (w★¹ᶜᶜᶠ * N²_local + Qᵇᵋ)
    ℓᵉ = clip(ϵˢᵖ * ℓᵉ)
    
    entraining = (Qᵇ > Qᵇᵋ) & (N²_local > 0) & (N²_above < 0)

    ℓ = ifelse(convecting, ℓᶜ,
        ifelse(entraining, ℓᵉ, zero(grid)))

    return ifelse(isnan(ℓ), zero(grid), ℓ)
end

@inline function convective_length_scaleᶜᶜᶜ(i, j, k, grid, closure, Cᶜ::Number, Cᵉ::Number, Cˢᵖ::Number,
                                            surface_buoyancy_flux, S², N², w★)

    Qᵇᵋ      = closure.minimum_convective_buoyancy_flux
    Qᵇ       = @inbounds surface_buoyancy_flux[i, j, 1]
    w★¹      = @inbounds w★[i, j, k]
    w★²      = @inbounds w★[i, j, k]^2
    w★³      = @inbounds w★[i, j, k]^3
    S²ᶜᶜᶜ    = ℑzᵃᵃᶜ(i, j, k, grid, S²)
    N²_local = ℑzᵃᵃᶜ(i, j, k, grid, N²)
    N²_above = ℑzᵃᵃᶜ(i, j, k+1, grid, N²)

    # "Convective length"
    # ℓᶜ ∼ boundary layer depth according to Deardorff scaling
    ℓᶜ = Cᶜ * w★³ / (Qᵇ + Qᵇᵋ)
    ℓᶜ = ifelse(isnan(ℓᶜ), zero(grid), ℓᶜ)

    # Figure out which mixing length applies
    convecting = (Qᵇ > Qᵇᵋ) & (N²_local < 0)

    # Model for shear-convection interaction
    Sp = sqrt(S²ᶜᶜᶜ) * w★² / (Qᵇ + Qᵇᵋ) # Sp = "Sheared convection number"
    ϵˢᵖ = 1 - Cˢᵖ * Sp               # ϵ = Sheared convection factor
    
    # Reduce convective and entraining mixing lengths by sheared convection factor
    # end ensure non-negativity
    ℓᶜ = clip(ϵˢᵖ * ℓᶜ)

    # "Entrainment length"
    # Ensures that w′b′ ~ Qᵇ at entrainment depth
    ℓᵉ = Cᵉ * Qᵇ / (w★¹ * N²_local + Qᵇᵋ)
    ℓᵉ = clip(ϵˢᵖ * ℓᵉ)
    
    entraining = (Qᵇ > Qᵇᵋ) & (N²_local > 0) & (N²_above < 0)

    ℓ = ifelse(convecting, ℓᶜ,
        ifelse(entraining, ℓᵉ, zero(grid)))

    return ifelse(isnan(ℓ), zero(grid), ℓ)
end

"""Piecewise linear function between 0 (when x < c) and 1 (when x - c > w)."""
@inline step(x, c, w) = max(zero(x), min(one(x), (x - c) / w))
@inline scale(Ri, σ⁻, σ⁺ , c, w) = σ⁻ + (σ⁺ - σ⁻) * step(Ri, c, w)

@inline function stability_functionᶜᶜᶠ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, S², N²)
    N²ᶜᶜᶠ = @inbounds N²[i, j, k]
    S²ᶜᶜᶠ = @inbounds S²[i, j, k]
    Ri = ifelse(N²ᶜᶜᶠ ≤ 0, zero(grid), N²ᶜᶜᶠ / S²ᶜᶜᶠ) 
    CRi⁰ = closure.mixing_length.CRi⁰
    CRiᵟ = closure.mixing_length.CRiᵟ
    return scale(Ri, Cˡᵒ, Cʰⁱ, CRi⁰, CRiᵟ)
end

@inline function stability_functionᶜᶜᶜ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, S², N²)
    N²ᶜᶜᶜ = ℑzᵃᵃᶠ(i, j, k, grid, N²)
    S²ᶜᶜᶜ = ℑzᵃᵃᶠ(i, j, k, grid, S²)
    Ri = ifelse(N²ᶜᶜᶜ ≤ 0, zero(grid), N²ᶜᶜᶜ / S²ᶜᶜᶜ) 
    CRi⁰ = closure.mixing_length.CRi⁰
    CRiᵟ = closure.mixing_length.CRiᵟ
    return scale(Ri, Cˡᵒ, Cʰⁱ, CRi⁰, CRiᵟ)
end

@inline function momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, S², N², w★)
    Cˡᵒ = closure.mixing_length.Cˡᵒu
    Cʰⁱ = closure.mixing_length.Cʰⁱu
    σ = stability_functionᶜᶜᶠ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, S², N²)

    ℓ★ = σ * stable_length_scaleᶜᶜᶠ(i, j, k, grid, closure, N², w★)
    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)

    H = total_depthᶜᶜᵃ(i, j, grid)

    return min(H, ℓ★)
end

@inline function momentum_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, S², N², w★)
    Cˡᵒ = closure.mixing_length.Cˡᵒu
    Cʰⁱ = closure.mixing_length.Cʰⁱu
    σ = stability_functionᶜᶜᶜ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, S², N²)

    ℓ★ = σ * stable_length_scaleᶜᶜᶜ(i, j, k, grid, closure, N², w★)
    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)

    H = total_depthᶜᶜᵃ(i, j, grid)

    return min(H, ℓ★)
end

@inline function tracer_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, surface_buoyancy_flux, S², N², w★)
    Cᶜ  = closure.mixing_length.Cᶜc
    Cᵉ  = closure.mixing_length.Cᵉc
    Cˢᵖ = closure.mixing_length.Cˢᵖ
    ℓʰ = convective_length_scaleᶜᶜᶠ(i, j, k, grid, closure, Cᶜ, Cᵉ, Cˢᵖ, surface_buoyancy_flux, S², N², w★)

    Cˡᵒ = closure.mixing_length.Cˡᵒc
    Cʰⁱ = closure.mixing_length.Cʰⁱc
    σ = stability_functionᶜᶜᶠ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, S², N²)
    ℓ★ = σ * stable_length_scaleᶜᶜᶠ(i, j, k, grid, closure, N², w★)

    ℓʰ = ifelse(isnan(ℓʰ), zero(grid), ℓʰ)
    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)
    ℓᶜ = max(ℓ★, ℓʰ)

    H = total_depthᶜᶜᵃ(i, j, grid)
    return min(H, ℓᶜ)
end

@inline function tracer_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, surface_buoyancy_flux, S², N², w★)
    Cᶜ  = closure.mixing_length.Cᶜc
    Cᵉ  = closure.mixing_length.Cᵉc
    Cˢᵖ = closure.mixing_length.Cˢᵖ
    ℓʰ = convective_length_scaleᶜᶜᶜ(i, j, k, grid, closure, Cᶜ, Cᵉ, Cˢᵖ, surface_buoyancy_flux, S², N², w★)

    Cˡᵒ = closure.mixing_length.Cˡᵒc
    Cʰⁱ = closure.mixing_length.Cʰⁱc
    σ = stability_functionᶜᶜᶜ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, S², N²)
    ℓ★ = σ * stable_length_scaleᶜᶜᶜ(i, j, k, grid, closure, N², w★)

    ℓʰ = ifelse(isnan(ℓʰ), zero(grid), ℓʰ)
    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)
    ℓᶜ = max(ℓ★, ℓʰ)

    H = total_depthᶜᶜᵃ(i, j, grid)

    return min(H, ℓᶜ)
end

@inline function TKE_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, surface_buoyancy_flux, S², N², w★)
    Cᶜ  = closure.mixing_length.Cᶜe
    Cᵉ  = closure.mixing_length.Cᵉe
    Cˢᵖ = closure.mixing_length.Cˢᵖ
    ℓʰ = convective_length_scaleᶜᶜᶠ(i, j, k, grid, closure, Cᶜ, Cᵉ, Cˢᵖ, surface_buoyancy_flux, S², N², w★)

    Cˡᵒ = closure.mixing_length.Cˡᵒe
    Cʰⁱ = closure.mixing_length.Cʰⁱe
    σ = stability_functionᶜᶜᶠ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, S², N²)
    ℓ★ = σ * stable_length_scaleᶜᶜᶜ(i, j, k, grid, closure,  N², w★)

    ℓʰ = ifelse(isnan(ℓʰ), zero(grid), ℓʰ)
    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)
    ℓᵉ = max(ℓ★, ℓʰ)

    H = total_depthᶜᶜᵃ(i, j, grid)
    return min(H, ℓᵉ)
end

Base.summary(::MixingLength) = "CATKEVerticalDiffusivities.MixingLength"

Base.show(io::IO, ml::MixingLength) =
    print(io, "CATKEVerticalDiffusivities.MixingLength parameters:", '\n',
              "    Cˢ:   $(ml.Cˢ)",   '\n',
              "    Cᵇ:   $(ml.Cᵇ)",   '\n',
              "    Cᶜc:  $(ml.Cᶜc)",  '\n',
              "    Cᶜe:  $(ml.Cᶜe)",  '\n',
              "    Cᵉc:  $(ml.Cᵉc)",  '\n',
              "    Cᵉe:  $(ml.Cᵉe)",  '\n',
              "    Cˡᵒu: $(ml.Cˡᵒu)", '\n',
              "    Cˡᵒc: $(ml.Cˡᵒc)", '\n',
              "    Cˡᵒe: $(ml.Cˡᵒe)", '\n',
              "    Cʰⁱu: $(ml.Cʰⁱu)", '\n',
              "    Cʰⁱc: $(ml.Cʰⁱc)", '\n',
              "    Cʰⁱe: $(ml.Cʰⁱe)", '\n',
              "    CRiᵟ: $(ml.CRiᵟ)", '\n',
              "    CRi⁰: $(ml.CRi⁰)")

