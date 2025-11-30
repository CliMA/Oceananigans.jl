using Oceananigans.Grids: AbstractGrid

const _ω̂₁ = 5/18
const _ω̂ₙ = 5/18
const _ε₂ = 1e-20

# Note: this can probably be generalized to include UpwindBiased
const BoundsPreservingWENO = WENO{<:Any, <:Any, <:Any, <:Tuple}

@inline div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, U, ::ZeroField) = zero(grid)

# Is this immersed-boundary safe without having to extend it in ImmersedBoundaries.jl? I think so... (velocity on immmersed boundaries is masked to 0)
@inline function div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, U, c)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, 1, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, 1, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, 1, U.w, c)

    return 1/Vᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

# Support for Flat directions
@inline bounded_tracer_flux_divergence_x(i, j, k, ::AbstractGrid{FT, Flat, TY, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TY, TZ} = zero(FT)
@inline bounded_tracer_flux_divergence_y(i, j, k, ::AbstractGrid{FT, TX, Flat, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TX, TZ} = zero(FT)
@inline bounded_tracer_flux_divergence_z(i, j, k, ::AbstractGrid{FT, TX, TY, Flat}, advection::BoundsPreservingWENO, args...) where {FT, TX, TY} = zero(FT)

@inline @inbounds function bounded_tracer_flux_divergence_x(i, j, k, grid, advection::BoundsPreservingWENO, ρ, u, c)
    c_min = advection.bounds[1]
    c_max = advection.bounds[2]

    c₊ᴸ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, LeftBias(),  c)
    c₊ᴿ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, RightBias(), c)
    c₋ᴸ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, LeftBias(),  c)
    c₋ᴿ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, RightBias(), c)

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    cᵢⱼ = c[i, j, k]
    p̃ = (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M = max(p̃, c₊ᴸ, c₋ᴿ)
    m = min(p̃, c₊ᴸ, c₋ᴿ)

    θ_max = abs((c_max - cᵢⱼ) / (M - cᵢⱼ + ε₂))
    θ_min = abs((c_min - cᵢⱼ) / (m - cᵢⱼ + ε₂))
    θ = min(θ_max, θ_min, one(grid))

    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    Ax_ρuc⁺ = ℑxᶠᵃᵃ(i+1, j, k, grid, ρ) * Axᶠᶜᶜ(i+1, j, k, grid) * upwind_biased_product(u[i+1, j, k], c₊ᴸ, c₊ᴿ)
    Ax_ρuc⁻ = ℑxᶠᵃᵃ(i,   j, k, grid, ρ) * Axᶠᶜᶜ(i,   j, k, grid) * upwind_biased_product(u[i,   j, k], c₋ᴸ, c₋ᴿ)

    return Ax_ρuc⁺ - Ax_ρuc⁻
end

@inline @inbounds function bounded_tracer_flux_divergence_y(i, j, k, grid, advection::BoundsPreservingWENO, ρ, v, c)
    c_min = advection.bounds[1]
    c_max = advection.bounds[2]

    c₊ᴸ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, LeftBias(),  c)
    c₊ᴿ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, RightBias(), c)
    c₋ᴸ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, LeftBias(),  c)
    c₋ᴿ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, RightBias(), c)

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    cᵢⱼ = c[i, j, k]
    p̃ = (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M = max(p̃, c₊ᴸ, c₋ᴿ)
    m = min(p̃, c₊ᴸ, c₋ᴿ)

    θ_max = abs((c_max - cᵢⱼ) / (M - cᵢⱼ + ε₂))
    θ_min = abs((c_min - cᵢⱼ) / (m - cᵢⱼ + ε₂))
    θ = min(θ_max, θ_min, one(grid))

    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    Ay_ρvc⁺ = ℑyᵃᶠᵃ(i, j+1, k, grid, ρ) * Ayᶜᶠᶜ(i, j+1, k, grid) * upwind_biased_product(v[i, j+1, k], c₊ᴸ, c₊ᴿ)
    Ay_ρvc⁻ = ℑyᵃᶠᵃ(i, j,   k, grid, ρ) * Ayᶜᶠᶜ(i, j,   k, grid) * upwind_biased_product(v[i, j,   k], c₋ᴸ, c₋ᴿ)

    return Ay_ρvc⁺ - Ay_ρvc⁻
end

@inline function bounded_tracer_flux_divergence_z(i, j, k, grid, advection::BoundsPreservingWENO, ρ, w, c)
    c_min = @inbounds advection.bounds[1]
    c_max = @inbounds advection.bounds[2]

    c₊ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, LeftBias(),  c)
    c₊ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, RightBias(), c)
    c₋ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, LeftBias(),  c)
    c₋ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, RightBias(), c)

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    cᵢⱼ = c[i, j, k]
    p̃ = (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M = max(p̃, c₊ᴸ, c₋ᴿ)
    m = min(p̃, c₊ᴸ, c₋ᴿ)

    θ_max = abs((c_max - cᵢⱼ) / (M - cᵢⱼ + ε₂))
    θ_min = abs((c_min - cᵢⱼ) / (m - cᵢⱼ + ε₂))
    θ = min(θ_max, θ_min, one(grid))

    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    Az_ρwc⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ) * Azᶜᶜᶠ(i, j, k+1, grid) * upwind_biased_product(w[i, j, k+1], c₊ᴸ, c₊ᴿ)
    Az_ρwc⁻ = ℑzᵃᵃᶠ(i, j, k,   grid, ρ) * Azᶜᶜᶠ(i, j, k,   grid) * upwind_biased_product(w[i, j, k],   c₋ᴸ, c₋ᴿ)

    return Az_ρwc⁺ - Az_ρwc⁻
end
