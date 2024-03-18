using ..TurbulenceClosures:
    wall_vertical_distanceᶜᶜᶠ,
    wall_vertical_distanceᶜᶜᶜ,
    depthᶜᶜᶠ,
    height_above_bottomᶜᶜᶠ,
    depthᶜᶜᶜ,
    height_above_bottomᶜᶜᶜ,
    total_depthᶜᶜᵃ

"""
    struct CATKEMixingLength{FT}

Contains mixing length parameters for CATKE vertical diffusivity.
"""
Base.@kwdef struct CATKEMixingLength{FT}
    Cˢ   :: FT = 0.814  # Surface distance coefficient for shear length scale
    Cᵇ   :: FT = Inf    # Bottom distance coefficient for shear length scaligm
    Cᶜc  :: FT = 10.0 #7.39   # Convective mixing length coefficient for tracers
    Cᶜe  :: FT = 7.35   # Convective mixing length coefficient for TKE
    Cᵉc  :: FT = 0.0 #1.29   # Convective penetration mixing length coefficient for tracers
    Cᵉe  :: FT = 0.0    # Convective penetration mixing length coefficient for TKE
    Cˢᵖ  :: FT = 0.584  # Sheared convective plume coefficient
    Cˡᵒu :: FT = 0.759  # Shear mixing length coefficient for momentum at low Ri
    Cʰⁱu :: FT = 0.816  # Shear mixing length coefficient for momentum at high Ri
    Cˡᵒc :: FT = 0.874  # Shear mixing length coefficient for tracers at low Ri
    Cʰⁱc :: FT = 0.348  # Shear mixing length coefficient for tracers at high Ri
    Cˡᵒe :: FT = 5.88   # Shear mixing length coefficient for TKE at low Ri
    Cʰⁱe :: FT = 7.39   # Shear mixing length coefficient for TKE at high Ri
    CRiᵟ :: FT = 0.620  # Stability function width 
    CRi⁰ :: FT = 0.233  # Stability function lower Ri
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

@inline function stratification_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, e, tracers, buoyancy)
    FT = eltype(grid)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = clip(N²) # so we can compute sqrt(N²⁺)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
    return ifelse(N²⁺ == 0, convert(FT, Inf), w★ / sqrt(N²⁺))
end

@inline function stratification_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    FT = eltype(grid)
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    N²⁺ = clip(N²) # so we can compute sqrt(N²⁺)
    w★ = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
    return ifelse(N²⁺ == 0, convert(FT, Inf), w★ / sqrt(N²⁺))
end

@inline function stable_length_scaleᶜᶜᶠ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    Cˢ = closure.mixing_length.Cˢ
    Cᵇ = closure.mixing_length.Cᵇ

    d_up   = Cˢ * depthᶜᶜᶠ(i, j, k, grid)
    d_down = Cᵇ * height_above_bottomᶜᶜᶠ(i, j, k, grid)
    d = min(d_up, d_down)

    ℓᴺ = stratification_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, e, tracers, buoyancy)

    ℓ = min(d, ℓᴺ)
    ℓ = ifelse(isnan(ℓ), d, ℓ)

    return ℓ
end

@inline function stable_length_scaleᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    Cˢ = closure.mixing_length.Cˢ
    Cᵇ = closure.mixing_length.Cᵇ
    d_up   = Cˢ * depthᶜᶜᶜ(i, j, k, grid)
    d_down = Cᵇ * height_above_bottomᶜᶜᶜ(i, j, k, grid)
    d = min(d_up, d_down)

    ℓᴺ = stratification_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)

    ℓ = min(d, ℓᴺ)
    ℓ = ifelse(isnan(ℓ), d, ℓ)

    return ℓ
end

@inline three_halves_tkeᶜᶜᶜ(i, j, k, grid, closure, e) = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)^3
@inline squared_tkeᶜᶜᶜ(i, j, k, grid, closure, e) = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)^2

@inline function convective_length_scaleᶜᶜᶠ(i, j, k, grid, closure, Cᶜ::Number, Cᵉ::Number, Cˢᵖ::Number,
                                            velocities, tracers, buoyancy, Jᵇ, hc)

    u = velocities.u
    v = velocities.v

    w★   = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    w★²  = ℑzᵃᵃᶠ(i, j, k, grid, squared_tkeᶜᶜᶜ, closure, tracers.e)
    S²   = shearᶜᶜᶠ(i, j, k, grid, u, v)
    N²   = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺¹ = ∂z_b(i, j, k+1, grid, buoyancy, tracers)
    Jᵇᵋ  = closure.minimum_convective_buoyancy_flux
    Jᵇˢ  = @inbounds Jᵇ[i, j, 1]

    # "Convective length"
    # ℓʰ ∼ boundary layer depth according to Deardorff scaling
    # Jᵇᵋ = closure.minimum_convective_buoyancy_flux
    # Jᵇⁱʲ  = @inbounds Jᵇ[i, j, 1]
    # w★³ = ℑzᵃᵃᶠ(i, j, k, grid, three_halves_tkeᶜᶜᶜ, closure, tracers.e)
    # ℓʰ = Cᶜ * w★³ / (Jᵇⁱʲ + Jᵇᵋ)
    # ℓʰ = ifelse(isnan(ℓʰ), zero(grid), ℓʰ)
    # H = total_depthᶜᶜᵃ(i, j, grid)
    # ℓʰ = max(ℓʰ, H)

    ℓʰ = @inbounds Cᶜ * hc[i, j, 1]

    # Model for shear-convection interaction
    Sp = sqrt(S²) * w★² / (Jᵇˢ + Jᵇᵋ) # Sp = "Sheared convection number"
    ϵˢᵖ = 1 - Cˢᵖ * Sp               # ϵ = Sheared convection factor
    
    # Reduce convective and entraining mixing lengths by sheared convection factor
    # end ensure non-negativity
    ℓʰ = clip(ϵˢᵖ * ℓʰ)

    # "Entrainment length"
    # Ensures that w′b′ ~ Jᵇ at entrainment depth
    ℓᵉ = Cᵉ * Jᵇˢ / (w★ * N² + Jᵇᵋ)
    ℓᵉ = clip(ϵˢᵖ * ℓᵉ)

    # Figure out which mixing length applies
    convecting = (Jᵇˢ > Jᵇᵋ) & (N² < 0)
    entraining = (Jᵇˢ > Jᵇᵋ) & (N² > 0) & (N²⁺¹ < 0)

    ℓ = ifelse(convecting, ℓʰ,
        ifelse(entraining, ℓᵉ, zero(grid)))

    return ifelse(isnan(ℓ), zero(grid), ℓ)
end

@inline function convective_length_scaleᶜᶜᶜ(i, j, k, grid, closure, Cᶜ::Number, Cᵉ::Number, Cˢᵖ::Number,
                                            velocities, tracers, buoyancy, Jᵇ, hc)

    u = velocities.u
    v = velocities.v

    w★   = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, tracers.e)
    w★²  = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, tracers.e)^2
    w★³  = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, tracers.e)^3
    S²   = shearᶜᶜᶜ(i, j, k, grid, u, v)
    N²   = ℑzᵃᵃᶜ(i, j, k,   grid, ∂z_b, buoyancy, tracers)
    N²⁺¹ = ℑzᵃᵃᶜ(i, j, k+1, grid, ∂z_b, buoyancy, tracers)
    Jᵇᵋ  = closure.minimum_convective_buoyancy_flux
    Jᵇˢ  = @inbounds Jᵇ[i, j, 1]

    # "Convective length"
    # ℓʰ ∼ boundary layer depth according to Deardorff scaling, w★ ~ (ℓ * Jᵇ)^(1/3)
    # wlim = (|z| * Jᵇ_min)^(1/3) ?
    # Jᵇ  = @inbounds surface_buoyancy_flux[i, j, 1]
    # ℓʰ = Cᶜ * w★³ / (Jᵇ + Jᵇᵋ)
    # ℓʰ = ifelse(isnan(ℓʰ), zero(grid), ℓʰ)
    # H = total_depthᶜᶜᵃ(i, j, grid)
    # ℓʰ = max(ℓʰ, H)

    ℓʰ = @inbounds Cᶜ * hc[i, j, 1]

    # Model for shear-convection interaction
    Sp = sqrt(S²) * w★² / (Jᵇˢ + Jᵇᵋ)    # Sp = "Sheared convection number"
    ϵˢᵖ = max(zero(grid), 1 - Cˢᵖ * Sp) # ϵ = Sheared convection factor
    
    # Reduce convective and entraining mixing lengths by sheared convection factor
    ℓʰ = ϵˢᵖ * ℓʰ

    # "Entrainment length"
    # Ensures that w′b′ ~ Jᵇ at entrainment depth
    ℓᵉ = Cᵉ * Jᵇˢ / (w★ * N² + Jᵇᵋ)
    ℓᵉ = ϵˢᵖ * ℓᵉ
    
    # Figure out which mixing length applies
    convecting = (Jᵇˢ > Jᵇᵋ) & (N² < 0)
    entraining = (Jᵇˢ > Jᵇᵋ) & (N² > 0) & (N²⁺¹ < 0)

    ℓ = ifelse(convecting, ℓʰ,
        ifelse(entraining, ℓᵉ, zero(grid)))

    return ifelse(isnan(ℓ), zero(grid), ℓ)
end

"""Piecewise linear function between 0 (when x < c) and 1 (when x - c > w)."""
@inline step(x, c, w) = max(zero(x), min(one(x), (x - c) / w))
@inline scale(Ri, σ⁻, σ⁺ , c, w) = σ⁻ + (σ⁺ - σ⁻) * step(Ri, c, w)

@inline function stability_functionᶜᶜᶠ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    CRi⁰ = closure.mixing_length.CRi⁰
    CRiᵟ = closure.mixing_length.CRiᵟ
    return scale(Ri, Cˡᵒ, Cʰⁱ, CRi⁰, CRiᵟ)
end

@inline function stability_functionᶜᶜᶜ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    CRi⁰ = closure.mixing_length.CRi⁰
    CRiᵟ = closure.mixing_length.CRiᵟ
    return scale(Ri, Cˡᵒ, Cʰⁱ, CRi⁰, CRiᵟ)
end

@inline function momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, Jᵇ, hc)
    Cˡᵒ = closure.mixing_length.Cˡᵒu
    Cʰⁱ = closure.mixing_length.Cʰⁱu
    ℓ★ = stable_length_scaleᶜᶜᶠ(i, j, k, grid, closure, tracers.e, velocities, tracers, buoyancy)

    σ = stability_functionᶜᶜᶠ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, velocities, tracers, buoyancy)
    return σ * ℓ★
end

@inline function tracer_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, Jᵇ, hc)
    Cᶜ  = closure.mixing_length.Cᶜc
    Cᵉ  = closure.mixing_length.Cᵉc
    Cˢᵖ = closure.mixing_length.Cˢᵖ
    ℓh = convective_length_scaleᶜᶜᶠ(i, j, k, grid, closure, Cᶜ, Cᵉ, Cˢᵖ, velocities, tracers, buoyancy, Jᵇ, hc)

    Cˡᵒ = closure.mixing_length.Cˡᵒc
    Cʰⁱ = closure.mixing_length.Cʰⁱc
    ℓ★ = stable_length_scaleᶜᶜᶠ(i, j, k, grid, closure, tracers.e, velocities, tracers, buoyancy)

    σ = stability_functionᶜᶜᶠ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, velocities, tracers, buoyancy)
    return σ * max(ℓ★, ℓh)
end

@inline function TKE_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, Jᵇ, hc)
    Cᶜ  = closure.mixing_length.Cᶜe
    Cᵉ  = closure.mixing_length.Cᵉe
    Cˢᵖ = closure.mixing_length.Cˢᵖ
    ℓh  = convective_length_scaleᶜᶜᶠ(i, j, k, grid, closure, Cᶜ, Cᵉ, Cˢᵖ, velocities, tracers, buoyancy, Jᵇ, hc)

    Cˡᵒ = closure.mixing_length.Cˡᵒe
    Cʰⁱ = closure.mixing_length.Cʰⁱe
    ℓ★ = stable_length_scaleᶜᶜᶠ(i, j, k, grid, closure, tracers.e, velocities, tracers, buoyancy)

    σ = stability_functionᶜᶜᶠ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, velocities, tracers, buoyancy)
    return σ * max(ℓ★, ℓh)
end

Base.summary(::CATKEMixingLength) = "CATKEMixingLength"

Base.show(io::IO, ml::CATKEMixingLength) =
    print(io, "CATKEMixingLength parameters:", '\n',
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

