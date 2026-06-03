using Oceananigans.Grids: AbstractGrid, SphericalShellGrid

const _ω̂₁ = 5/18
const _ω̂ₙ = 5/18
const _ε₂ = 1e-20

# Note: this can probably be generalized to include UpwindBiased
const BoundsPreservingWENO = WENO{<:Any, <:Any, <:Any, <:Tuple}

@inline div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, U, ::ZeroField) = zero(grid)

@inline function _bounds_preserving_spherical_shell_advective_tracer_flux_x(i, j, k, grid, advection, U, c)
    c_min = @inbounds advection.bounds[1]
    c_max = @inbounds advection.bounds[2]

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    volume_flux = spherical_shell_horizontal_tracer_flux_u(U, i, j, k)

    c₊ᴸ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_xᶠᵃᵃ(i-1, j, k, grid, advection, RightBias, c)
    cᵢ₋₁ⱼ = @inbounds c[i-1, j, k]
    p̃₊ = (cᵢ₋₁ⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M₊ = max(p̃₊, c₊ᴸ, c₋ᴿ)
    m₊ = min(p̃₊, c₊ᴸ, c₋ᴿ)
    θ₊max = abs((c_max - cᵢ₋₁ⱼ) / (M₊ - cᵢ₋₁ⱼ + ε₂))
    θ₊min = abs((c_min - cᵢ₋₁ⱼ) / (m₊ - cᵢ₋₁ⱼ + ε₂))
    θ₊ = min(θ₊max, θ₊min, one(grid))
    limited_positive_state = θ₊ * (c₊ᴸ - cᵢ₋₁ⱼ) + cᵢ₋₁ⱼ

    c₊ᴸ₂ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, LeftBias,  c)
    c₋ᴿ₂ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, RightBias, c)
    cᵢⱼ = @inbounds c[i, j, k]
    p̃₋ = (cᵢⱼ - ω̂₁ * c₋ᴿ₂ - ω̂ₙ * c₊ᴸ₂) / (1 - 2ω̂₁)
    M₋ = max(p̃₋, c₊ᴸ₂, c₋ᴿ₂)
    m₋ = min(p̃₋, c₊ᴸ₂, c₋ᴿ₂)
    θ₋max = abs((c_max - cᵢⱼ) / (M₋ - cᵢⱼ + ε₂))
    θ₋min = abs((c_min - cᵢⱼ) / (m₋ - cᵢⱼ + ε₂))
    θ₋ = min(θ₋max, θ₋min, one(grid))
    limited_negative_state = θ₋ * (c₋ᴿ₂ - cᵢⱼ) + cᵢⱼ

    return ifelse(volume_flux > zero(FT),
                  volume_flux * limited_positive_state,
                  volume_flux * limited_negative_state)
end

@inline advective_tracer_flux_x(i, j, k, grid::SphericalShellGrid, advection::BoundsPreservingWENO, U, c) =
    _bounds_preserving_spherical_shell_advective_tracer_flux_x(i, j, k, grid, advection, U, c)

@inline advective_tracer_flux_x(i, j, k, grid::OHPSG, advection::BoundsPreservingWENO, U, c) =
    _bounds_preserving_spherical_shell_advective_tracer_flux_x(i, j, k, grid, advection, U, c)

@inline function advective_tracer_flux_y(i, j, k, grid::SphericalShellGrid, advection::BoundsPreservingWENO, U, c)
    c_min = @inbounds advection.bounds[1]
    c_max = @inbounds advection.bounds[2]

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    volume_flux = spherical_shell_horizontal_tracer_flux_v(U, i, j, k)

    c₊ᴸ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_yᵃᶠᵃ(i, j-1, k, grid, advection, RightBias, c)
    cᵢⱼ₋₁ = @inbounds c[i, j-1, k]
    p̃₊ = (cᵢⱼ₋₁ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M₊ = max(p̃₊, c₊ᴸ, c₋ᴿ)
    m₊ = min(p̃₊, c₊ᴸ, c₋ᴿ)
    θ₊max = abs((c_max - cᵢⱼ₋₁) / (M₊ - cᵢⱼ₋₁ + ε₂))
    θ₊min = abs((c_min - cᵢⱼ₋₁) / (m₊ - cᵢⱼ₋₁ + ε₂))
    θ₊ = min(θ₊max, θ₊min, one(grid))
    limited_positive_state = θ₊ * (c₊ᴸ - cᵢⱼ₋₁) + cᵢⱼ₋₁

    c₊ᴸ₂ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, LeftBias,  c)
    c₋ᴿ₂ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, RightBias, c)
    cᵢⱼ = @inbounds c[i, j, k]
    p̃₋ = (cᵢⱼ - ω̂₁ * c₋ᴿ₂ - ω̂ₙ * c₊ᴸ₂) / (1 - 2ω̂₁)
    M₋ = max(p̃₋, c₊ᴸ₂, c₋ᴿ₂)
    m₋ = min(p̃₋, c₊ᴸ₂, c₋ᴿ₂)
    θ₋max = abs((c_max - cᵢⱼ) / (M₋ - cᵢⱼ + ε₂))
    θ₋min = abs((c_min - cᵢⱼ) / (m₋ - cᵢⱼ + ε₂))
    θ₋ = min(θ₋max, θ₋min, one(grid))
    limited_negative_state = θ₋ * (c₋ᴿ₂ - cᵢⱼ) + cᵢⱼ

    return ifelse(volume_flux > zero(FT),
                  volume_flux * limited_positive_state,
                  volume_flux * limited_negative_state)
end

@inline _nonorthogonal_advective_tracer_flux_x(i, j, k, grid::SphericalShellGrid, advection::BoundsPreservingWENO, U, c) =
    advective_tracer_flux_x(i, j, k, grid, advection, U, c)

@inline _nonorthogonal_advective_tracer_flux_y(i, j, k, grid::SphericalShellGrid, advection::BoundsPreservingWENO, U, c) =
    advective_tracer_flux_y(i, j, k, grid, advection, U, c)

# Is this immersed-boundary safe without having to extend it in ImmersedBoundaries.jl? I think so... (velocity on immmersed boundaries is masked to 0)
# For bounds preserving advection, we need fluxes at both cell-faces to compute the flux on one face.
# So we extend div_Uc in order to compute the fluxes at i and i+1 in one go and avoid recomputation.
@inline function div_Uc(i, j, k, grid, advection::BoundsPreservingWENO, U, c)
    u = u_velocity(U)
    v = v_velocity(U)
    w = w_velocity(U)

    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, 1, u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, 1, v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, 1, w, c)

    return 1/Vᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

@inline function div_Uc(i, j, k, grid::SphericalShellGrid, advection::BoundsPreservingWENO, U, c)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, 1, U, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, 1, U, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, 1, w_velocity(U), c)

    return 1/Vᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

# Support for Flat directions
@inline bounded_tracer_flux_divergence_x(i, j, k, ::AbstractGrid{FT, Flat, TY, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TY, TZ} = zero(FT)
@inline bounded_tracer_flux_divergence_y(i, j, k, ::AbstractGrid{FT, TX, Flat, TZ}, advection::BoundsPreservingWENO, args...) where {FT, TX, TZ} = zero(FT)
@inline bounded_tracer_flux_divergence_z(i, j, k, ::AbstractGrid{FT, TX, TY, Flat}, advection::BoundsPreservingWENO, args...) where {FT, TX, TY} = zero(FT)

@inline function bounded_tracer_flux_divergence_x(i, j, k, grid, advection::BoundsPreservingWENO, ρ, u, c)
    c_min = @inbounds advection.bounds[1]
    c_max = @inbounds advection.bounds[2]

    c₊ᴸ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, LeftBias,  c)
    c₊ᴿ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, RightBias, c)
    c₋ᴸ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, RightBias, c)

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    cᵢⱼ = @inbounds c[i, j, k]
    p̃ = (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M = max(p̃, c₊ᴸ, c₋ᴿ)
    m = min(p̃, c₊ᴸ, c₋ᴿ)

    θ_max = abs((c_max - cᵢⱼ) / (M - cᵢⱼ + ε₂))
    θ_min = abs((c_min - cᵢⱼ) / (m - cᵢⱼ + ε₂))
    θ = min(θ_max, θ_min, one(grid))

    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    u⁺ = @inbounds u[i+1, j, k]
    u⁻ = @inbounds u[i,   j, k]
    Ax_ρuc⁺ = ℑxᶠᵃᵃ(i+1, j, k, grid, ρ) * Axᶠᶜᶜ(i+1, j, k, grid) * upwind_biased_product(u⁺, c₊ᴸ, c₊ᴿ)
    Ax_ρuc⁻ = ℑxᶠᵃᵃ(i,   j, k, grid, ρ) * Axᶠᶜᶜ(i,   j, k, grid) * upwind_biased_product(u⁻, c₋ᴸ, c₋ᴿ)

    return Ax_ρuc⁺ - Ax_ρuc⁻
end

@inline function bounded_tracer_flux_divergence_x(i, j, k, grid::SphericalShellGrid, advection::BoundsPreservingWENO, ρ, U, c)
    c_min = @inbounds advection.bounds[1]
    c_max = @inbounds advection.bounds[2]

    c₊ᴸ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, LeftBias,  c)
    c₊ᴿ = _biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, RightBias, c)
    c₋ᴸ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, RightBias, c)

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    cᵢⱼ = @inbounds c[i, j, k]
    p̃ = (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M = max(p̃, c₊ᴸ, c₋ᴿ)
    m = min(p̃, c₊ᴸ, c₋ᴿ)

    θ_max = abs((c_max - cᵢⱼ) / (M - cᵢⱼ + ε₂))
    θ_min = abs((c_min - cᵢⱼ) / (m - cᵢⱼ + ε₂))
    θ = min(θ_max, θ_min, one(grid))

    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    u⁺ = spherical_shell_horizontal_tracer_flux_u(U, i+1, j, k)
    u⁻ = spherical_shell_horizontal_tracer_flux_u(U, i,   j, k)
    ρuc⁺ = ℑxᶠᵃᵃ(i+1, j, k, grid, ρ) * upwind_biased_product(u⁺, c₊ᴸ, c₊ᴿ)
    ρuc⁻ = ℑxᶠᵃᵃ(i,   j, k, grid, ρ) * upwind_biased_product(u⁻, c₋ᴸ, c₋ᴿ)

    return ρuc⁺ - ρuc⁻
end

@inline function bounded_tracer_flux_divergence_y(i, j, k, grid, advection::BoundsPreservingWENO, ρ, v, c)
    c_min = @inbounds advection.bounds[1]
    c_max = @inbounds advection.bounds[2]

    c₊ᴸ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, LeftBias,  c)
    c₊ᴿ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, RightBias, c)
    c₋ᴸ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, RightBias, c)

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    cᵢⱼ = @inbounds c[i, j, k]
    p̃ = (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M = max(p̃, c₊ᴸ, c₋ᴿ)
    m = min(p̃, c₊ᴸ, c₋ᴿ)

    θ_max = abs((c_max - cᵢⱼ) / (M - cᵢⱼ + ε₂))
    θ_min = abs((c_min - cᵢⱼ) / (m - cᵢⱼ + ε₂))
    θ = min(θ_max, θ_min, one(grid))

    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    v⁺ = @inbounds v[i, j+1, k]
    v⁻ = @inbounds v[i, j,   k]
    Ay_ρvc⁺ = ℑyᵃᶠᵃ(i, j+1, k, grid, ρ) * Ayᶜᶠᶜ(i, j+1, k, grid) * upwind_biased_product(v⁺, c₊ᴸ, c₊ᴿ)
    Ay_ρvc⁻ = ℑyᵃᶠᵃ(i, j,   k, grid, ρ) * Ayᶜᶠᶜ(i, j,   k, grid) * upwind_biased_product(v⁻, c₋ᴸ, c₋ᴿ)

    return Ay_ρvc⁺ - Ay_ρvc⁻
end

@inline function bounded_tracer_flux_divergence_y(i, j, k, grid::SphericalShellGrid, advection::BoundsPreservingWENO, ρ, U, c)
    c_min = @inbounds advection.bounds[1]
    c_max = @inbounds advection.bounds[2]

    c₊ᴸ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, LeftBias,  c)
    c₊ᴿ = _biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, RightBias, c)
    c₋ᴸ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, RightBias, c)

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    cᵢⱼ = @inbounds c[i, j, k]
    p̃ = (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M = max(p̃, c₊ᴸ, c₋ᴿ)
    m = min(p̃, c₊ᴸ, c₋ᴿ)

    θ_max = abs((c_max - cᵢⱼ) / (M - cᵢⱼ + ε₂))
    θ_min = abs((c_min - cᵢⱼ) / (m - cᵢⱼ + ε₂))
    θ = min(θ_max, θ_min, one(grid))

    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    v⁺ = spherical_shell_horizontal_tracer_flux_v(U, i, j+1, k)
    v⁻ = spherical_shell_horizontal_tracer_flux_v(U, i, j,   k)
    ρvc⁺ = ℑyᵃᶠᵃ(i, j+1, k, grid, ρ) * upwind_biased_product(v⁺, c₊ᴸ, c₊ᴿ)
    ρvc⁻ = ℑyᵃᶠᵃ(i, j,   k, grid, ρ) * upwind_biased_product(v⁻, c₋ᴸ, c₋ᴿ)

    return ρvc⁺ - ρvc⁻
end

@inline function bounded_tracer_flux_divergence_z(i, j, k, grid, advection::BoundsPreservingWENO, ρ, w, c)
    c_min = @inbounds advection.bounds[1]
    c_max = @inbounds advection.bounds[2]

    c₊ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, LeftBias,  c)
    c₊ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, RightBias, c)
    c₋ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, RightBias, c)

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    cᵢⱼ = @inbounds c[i, j, k]
    p̃ = (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M = max(p̃, c₊ᴸ, c₋ᴿ)
    m = min(p̃, c₊ᴸ, c₋ᴿ)

    θ_max = abs((c_max - cᵢⱼ) / (M - cᵢⱼ + ε₂))
    θ_min = abs((c_min - cᵢⱼ) / (m - cᵢⱼ + ε₂))
    θ = min(θ_max, θ_min, one(grid))

    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    w⁺ = @inbounds w[i, j, k+1]
    w⁻ = @inbounds w[i, j, k]
    Az_ρwc⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ) * Azᶜᶜᶠ(i, j, k+1, grid) * upwind_biased_product(w⁺, c₊ᴸ, c₊ᴿ)
    Az_ρwc⁻ = ℑzᵃᵃᶠ(i, j, k,   grid, ρ) * Azᶜᶜᶠ(i, j, k,   grid) * upwind_biased_product(w⁻, c₋ᴸ, c₋ᴿ)

    return Az_ρwc⁺ - Az_ρwc⁻
end
