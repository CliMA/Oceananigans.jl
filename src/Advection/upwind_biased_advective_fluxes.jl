#####
##### Momentum and tracer advective flux operators for upwind-biased advection schemes
#####
##### See topologically_conditional_interpolation.jl for an explanation of the underscore-prepended
##### functions _symmetric_interpolate_*, _left_biased_interpolate_*, and _right_biased_interpolate_*.
#####

const UpwindScheme = AbstractUpwindBiasedAdvectionScheme

@inline upwind_biased_product(ũ, ψᴸ, ψᴿ) = ((ũ + abs(ũ)) * ψᴸ + (ũ - abs(ũ)) * ψᴿ) / 2

#####
##### Momentum advection operators
#####
##### Note the convention "advective_momentum_flux_AB" corresponds to the advection _of_ B _by_ A.
#####

@enum Bias LeftBias RightBias

@inline bias(u::Number) = ifelse(u > 0, LeftBias, RightBias)

@inline function advective_momentum_flux_Uu(i, j, k, grid, scheme::UpwindScheme, U, u)

    ũ  = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, Ax_qᶠᶜᶜ, U)
    uᴿ =    _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, bias(ũ), u)

    return ũ * uᴿ
end

@inline function advective_momentum_flux_Uu(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, u)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    ũ = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, covariant_to_volume_flux_uᶠᶜᶜ, u_component, v_component)
    uᴿ = _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, bias(ũ), u)

    return ũ * uᴿ
end

@inline function advective_momentum_flux_Vu(i, j, k, grid, scheme::UpwindScheme, V, u)

    ṽ  = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, Ay_qᶜᶠᶜ, V)
    uᴿ =    _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias(ṽ), u)

    return ṽ * uᴿ
end

@inline function advective_momentum_flux_Vu(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, u)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    ṽ = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, covariant_to_volume_flux_vᶜᶠᶜ, u_component, v_component)
    uᴿ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias(ṽ), u)

    return ṽ * uᴿ
end

@inline function advective_momentum_flux_Wu(i, j, k, grid, scheme::UpwindScheme, W, u)

    w̃  = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, Az_qᶜᶜᶠ, W)
    uᴿ =    _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, bias(w̃), u)

    return w̃ * uᴿ
end

@inline function advective_momentum_flux_Uv(i, j, k, grid, scheme::UpwindScheme, U, v)

    ũ  = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, Ax_qᶠᶜᶜ, U)
    vᴿ =    _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias(ũ), v)

    return ũ * vᴿ
end

@inline function advective_momentum_flux_Uv(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, v)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    ũ = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, covariant_to_volume_flux_uᶠᶜᶜ, u_component, v_component)
    vᴿ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias(ũ), v)

    return ũ * vᴿ
end

@inline function advective_momentum_flux_Vv(i, j, k, grid, scheme::UpwindScheme, V, v)

    ṽ  = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, Ay_qᶜᶠᶜ, V)
    vᴿ =    _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, bias(ṽ), v)

    return ṽ * vᴿ
end

@inline function advective_momentum_flux_Vv(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, v)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    ṽ = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, covariant_to_volume_flux_vᶜᶠᶜ, u_component, v_component)
    vᴿ = _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, bias(ṽ), v)

    return ṽ * vᴿ
end

@inline function advective_momentum_flux_Wv(i, j, k, grid, scheme::UpwindScheme, W, v)

    w̃  = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, Az_qᶜᶜᶠ, W)
    vᴿ =    _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, bias(w̃), v)

    return w̃ * vᴿ
end

@inline function advective_momentum_flux_Uw(i, j, k, grid, scheme::UpwindScheme, U, w)

    ũ  = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, Ax_qᶠᶜᶜ, U)
    wᴿ =    _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias(ũ), w)

    return ũ * wᴿ
end

@inline function advective_momentum_flux_Uw(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, w)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    ũ = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, covariant_to_volume_flux_uᶠᶜᶜ, u_component, v_component)
    wᴿ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias(ũ), w)

    return ũ * wᴿ
end

@inline function advective_momentum_flux_Vw(i, j, k, grid, scheme::UpwindScheme, V, w)

    ṽ  = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, Ay_qᶜᶠᶜ, V)
    wᴿ =    _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias(ṽ), w)

    return ṽ * wᴿ
end

@inline function advective_momentum_flux_Vw(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, w)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    ṽ = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, covariant_to_volume_flux_vᶜᶠᶜ, u_component, v_component)
    wᴿ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias(ṽ), w)

    return ṽ * wᴿ
end

@inline function advective_momentum_flux_Ww(i, j, k, grid, scheme::UpwindScheme, W, w)

    w̃  = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, Az_qᶜᶜᶠ, W)
    wᴿ =    _biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, bias(w̃), w)

    return w̃ * wᴿ
end

#####
##### Tracer advection operators
#####

@inline function advective_tracer_flux_x(i, j, k, grid, scheme::UpwindScheme, U, c)

    @inbounds ũ = U[i, j, k]
    cᴿ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias(ũ), c)

    return Axᶠᶜᶜ(i, j, k, grid) * ũ * cᴿ
end

@inline function advective_tracer_flux_y(i, j, k, grid, scheme::UpwindScheme, V, c)

    @inbounds ṽ = V[i, j, k]
    cᴿ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias(ṽ), c)

    return Ayᶜᶠᶜ(i, j, k, grid) * ṽ * cᴿ
end

@inline function advective_tracer_flux_z(i, j, k, grid, scheme::UpwindScheme, W, c)

    @inbounds w̃ = W[i, j, k]
    cᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, bias(w̃), c)

    return Azᶜᶜᶠ(i, j, k, grid) * w̃ * cᴿ
end

@inline function advective_tracer_flux_x(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, c)
    ũ = spherical_shell_horizontal_tracer_flux_u(U, i, j, k)
    cᴿ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias(ũ), c)
    return ũ * cᴿ
end

@inline function octahealpix_polar_xface_biased_tracer_state(i, j, k, grid, c, u_bias)
    FT = eltype(grid)
    half = convert(FT, 1//2)
    upwind_i = ifelse(u_bias == LeftBias, i - 1, i)
    adjacent_j = ifelse(j == 1, 2, grid.Ny - 1)
    folded_upwind_i = octahealpix_folded_polar_i(upwind_i, grid)

    @inbounds pole_state = half * (c[upwind_i, j, k] +
                                   c[folded_upwind_i, j, k])

    @inbounds outer_edge_state = half * (c[upwind_i, j, k] +
                                         c[upwind_i, adjacent_j, k])

    return half * (pole_state + outer_edge_state)
end

@inline function advective_tracer_flux_x(i, j, k, grid::OHPSG, scheme::UpwindScheme, U, c)
    ũ = spherical_shell_horizontal_tracer_flux_u(U, i, j, k)
    u_bias = bias(ũ)
    regular_cᴿ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, u_bias, c)
    polar_row = (j == 1) | (j == grid.Ny)
    polar_cᴿ = octahealpix_polar_xface_biased_tracer_state(i, j, k, grid, c, u_bias)
    weight = convert(eltype(grid), 3//4)
    blended_cᴿ = regular_cᴿ + weight * (polar_cᴿ - regular_cᴿ)
    cᴿ = ifelse(polar_row, blended_cᴿ, regular_cᴿ)
    return ũ * cᴿ
end

@inline function advective_tracer_flux_y(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, c)
    ṽ = spherical_shell_horizontal_tracer_flux_v(U, i, j, k)
    cᴿ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias(ṽ), c)
    return ṽ * cᴿ
end
using Oceananigans.Grids: SphericalShellGrid
using Oceananigans.Operators: covariant_to_volume_flux_uᶠᶜᶜ, covariant_to_volume_flux_vᶜᶠᶜ
