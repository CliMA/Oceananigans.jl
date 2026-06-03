using Oceananigans.Grids: SphericalShellGrid
using Oceananigans.Fields: ZeroField
using Oceananigans.Operators: δzᵃᵃᶠ

#####
##### Momentum advection operators
#####

# Alternate names for advective fluxes
@inline _advective_momentum_flux_Uu(i, j, k, grid, scheme, U, u) = advective_momentum_flux_Uu(i, j, k, grid, scheme, U, u)
@inline _advective_momentum_flux_Vu(i, j, k, grid, scheme, V, u) = advective_momentum_flux_Vu(i, j, k, grid, scheme, V, u)
@inline _advective_momentum_flux_Wu(i, j, k, grid, scheme, W, u) = advective_momentum_flux_Wu(i, j, k, grid, scheme, W, u)

@inline _advective_momentum_flux_Uv(i, j, k, grid, scheme, U, v) = advective_momentum_flux_Uv(i, j, k, grid, scheme, U, v)
@inline _advective_momentum_flux_Vv(i, j, k, grid, scheme, V, v) = advective_momentum_flux_Vv(i, j, k, grid, scheme, V, v)
@inline _advective_momentum_flux_Wv(i, j, k, grid, scheme, W, v) = advective_momentum_flux_Wv(i, j, k, grid, scheme, W, v)

@inline _advective_momentum_flux_Uw(i, j, k, grid, scheme, U, w) = advective_momentum_flux_Uw(i, j, k, grid, scheme, U, w)
@inline _advective_momentum_flux_Vw(i, j, k, grid, scheme, V, w) = advective_momentum_flux_Vw(i, j, k, grid, scheme, V, w)
@inline _advective_momentum_flux_Ww(i, j, k, grid, scheme, W, w) = advective_momentum_flux_Ww(i, j, k, grid, scheme, W, w)

const ZeroU = NamedTuple{(:u, :v, :w), Tuple{ZeroField, ZeroField, ZeroField}}

# Compiler hints
@inline div_𝐯u(i, j, k, grid, advection, ::ZeroU, u) = zero(grid)
@inline div_𝐯v(i, j, k, grid, advection, ::ZeroU, v) = zero(grid)
@inline div_𝐯w(i, j, k, grid, advection, ::ZeroU, w) = zero(grid)

@inline div_𝐯u(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)
@inline div_𝐯v(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)
@inline div_𝐯w(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)

@inline div_𝐯u(i, j, k, grid, advection, ::ZeroU, ::ZeroField) = zero(grid)
@inline div_𝐯v(i, j, k, grid, advection, ::ZeroU, ::ZeroField) = zero(grid)
@inline div_𝐯w(i, j, k, grid, advection, ::ZeroU, ::ZeroField) = zero(grid)

"""
    div_𝐯u(i, j, k, grid, advection, U, u)

Calculate the advection of momentum in the ``x``-direction using the conservative form, ``𝛁⋅(𝐯 u)``,

```
1/Vᵘ * [δxᶠᵃᵃ(ℑxᶜᵃᵃ(Ax * u) * ℑxᶜᵃᵃ(u)) + δy_fca(ℑxᶠᵃᵃ(Ay * v) * ℑyᵃᶠᵃ(u)) + δz_fac(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(u))]
```

which ends up at the location `fcc`.
"""
@inline function div_𝐯u(i, j, k, grid, advection, U, u)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    w_component = w_velocity(U)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uu, advection, u_component, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vu, advection, v_component, u) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, advection, w_component, u))
end

@inline function div_𝐯u(i, j, k, grid::SphericalShellGrid, advection, U, u)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uu, advection, U, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vu, advection, U, u) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, advection, w_velocity(U), u))
end

"""
    div_𝐯v(i, j, k, grid, advection, U, v)

Calculate the advection of momentum in the ``y``-direction using the conservative form, ``𝛁⋅(𝐯 v)``,

```
1/Vʸ * [δx_cfa(ℑyᵃᶠᵃ(Ax * u) * ℑxᶠᵃᵃ(v)) + δyᵃᶠᵃ(ℑyᵃᶜᵃ(Ay * v) * ℑyᵃᶜᵃ(v)) + δz_afc(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(w))]
```

which ends up at the location `cfc`.
"""
@inline function div_𝐯v(i, j, k, grid, advection, U, v)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    w_component = w_velocity(U)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uv, advection, u_component, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, _advective_momentum_flux_Vv, advection, v_component, v)    +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, advection, w_component, v))
end

@inline function div_𝐯v(i, j, k, grid::SphericalShellGrid, advection, U, v)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uv, advection, U, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, _advective_momentum_flux_Vv, advection, U, v) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, advection, w_velocity(U), v))
end

"""
    div_𝐯w(i, j, k, grid, advection, U, w)

Calculate the advection of momentum in the ``z``-direction using the conservative form, ``𝛁⋅(𝐯 w)``,

```
1/Vʷ * [δx_caf(ℑzᵃᵃᶠ(Ax * u) * ℑxᶠᵃᵃ(w)) + δy_acf(ℑzᵃᵃᶠ(Ay * v) * ℑyᵃᶠᵃ(w)) + δzᵃᵃᶠ(ℑzᵃᵃᶜ(Az * w) * ℑzᵃᵃᶜ(w))]
```
which ends up at the location `ccf`.
"""
@inline function div_𝐯w(i, j, k, grid, advection, U, w)
    u_component = u_velocity(U)
    v_component = v_velocity(U)
    w_component = w_velocity(U)
    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uw, advection, u_component, w) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vw, advection, v_component, w) +
                                    δzᵃᵃᶠ(i, j, k, grid, _advective_momentum_flux_Ww, advection, w_component, w))
end

@inline function div_𝐯w(i, j, k, grid::SphericalShellGrid, advection, U, w)
    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uw, advection, U, w) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vw, advection, U, w) +
                                    δzᵃᵃᶠ(i, j, k, grid, _advective_momentum_flux_Ww, advection, w_velocity(U), w))
end

@inline div_𝐯u(i, j, k, grid::SphericalShellGrid, advection, ::ZeroU, u) = zero(grid)
@inline div_𝐯v(i, j, k, grid::SphericalShellGrid, advection, ::ZeroU, v) = zero(grid)
@inline div_𝐯w(i, j, k, grid::SphericalShellGrid, advection, ::ZeroU, w) = zero(grid)

@inline div_𝐯u(i, j, k, grid::SphericalShellGrid, advection, U, ::ZeroField) = zero(grid)
@inline div_𝐯v(i, j, k, grid::SphericalShellGrid, advection, U, ::ZeroField) = zero(grid)
@inline div_𝐯w(i, j, k, grid::SphericalShellGrid, advection, U, ::ZeroField) = zero(grid)

@inline div_𝐯u(i, j, k, grid::SphericalShellGrid, advection, ::ZeroU, ::ZeroField) = zero(grid)
@inline div_𝐯v(i, j, k, grid::SphericalShellGrid, advection, ::ZeroU, ::ZeroField) = zero(grid)
@inline div_𝐯w(i, j, k, grid::SphericalShellGrid, advection, ::ZeroU, ::ZeroField) = zero(grid)

#####
##### Momentum advection by stored SphericalShellGrid transport fluxes
#####

@inline projected_transport_momentum_scheme(grid, advection::VectorInvariant) =
    projected_transport_momentum_scheme(grid, advection.divergence_scheme)

@inline projected_transport_momentum_scheme(grid, advection::AbstractUpwindBiasedAdvectionScheme) = advection
@inline projected_transport_momentum_scheme(grid, advection) = Centered(eltype(grid))

@inline _transport_momentum_flux_Uu(i, j, k, grid, scheme, U, u) =
    transport_momentum_flux_Uu(i, j, k, grid, scheme, U, u)

@inline _transport_momentum_flux_Vu(i, j, k, grid, scheme, U, u) =
    transport_momentum_flux_Vu(i, j, k, grid, scheme, U, u)

@inline _transport_momentum_flux_Uv(i, j, k, grid, scheme, U, v) =
    transport_momentum_flux_Uv(i, j, k, grid, scheme, U, v)

@inline _transport_momentum_flux_Vv(i, j, k, grid, scheme, U, v) =
    transport_momentum_flux_Vv(i, j, k, grid, scheme, U, v)

@inline function transport_momentum_flux_Uu(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, u)
    ũ = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u_velocity(U))
    uᶜᶜᶜ = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)
    return ũ * uᶜᶜᶜ
end

@inline function transport_momentum_flux_Vu(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, u)
    ṽ = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v_velocity(U))
    uᶠᶠᶜ = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
    return ṽ * uᶠᶠᶜ
end

@inline function transport_momentum_flux_Vu(i, j, k, grid::OHPSG, scheme::CenteredScheme, U, u)
    ṽ = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v_velocity(U))
    uᶠᶠᶜ = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
    regular_flux = ṽ * uᶠᶠᶜ
    polar_fold = (j == 1) | (j == grid.Ny + 1)
    return ifelse(polar_fold, zero(grid), regular_flux)
end

@inline function transport_momentum_flux_Uv(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, v)
    ũ = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u_velocity(U))
    vᶠᶠᶜ = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
    return ũ * vᶠᶠᶜ
end

@inline function transport_momentum_flux_Vv(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, v)
    ṽ = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v_velocity(U))
    vᶜᶜᶜ = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
    return ṽ * vᶜᶜᶜ
end

@inline function transport_momentum_flux_Vv(i, j, k, grid::OHPSG, scheme::CenteredScheme, U, v)
    ṽ = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v_velocity(U))
    vᶜᶜᶜ = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
    regular_flux = ṽ * vᶜᶜᶜ
    polar_adjacent_row = (j == 1) | (j == grid.Ny)
    return ifelse(polar_adjacent_row, zero(grid), regular_flux)
end

@inline function transport_momentum_flux_Uu(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, u)
    ũ = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u_velocity(U))
    uᴿ = _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, bias(ũ), u)
    return ũ * uᴿ
end

@inline function transport_momentum_flux_Vu(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, u)
    ṽ = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v_velocity(U))
    uᴿ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias(ṽ), u)
    return ṽ * uᴿ
end

@inline function transport_momentum_flux_Vu(i, j, k, grid::OHPSG, scheme::UpwindScheme, U, u)
    ṽ = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v_velocity(U))
    uᴿ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias(ṽ), u)
    regular_flux = ṽ * uᴿ
    polar_fold = (j == 1) | (j == grid.Ny + 1)
    return ifelse(polar_fold, zero(grid), regular_flux)
end

@inline function transport_momentum_flux_Uv(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, v)
    ũ = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u_velocity(U))
    vᴿ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias(ũ), v)
    return ũ * vᴿ
end

@inline function transport_momentum_flux_Vv(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, v)
    ṽ = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v_velocity(U))
    vᴿ = _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, bias(ṽ), v)
    return ṽ * vᴿ
end

@inline function transport_momentum_flux_Vv(i, j, k, grid::OHPSG, scheme::UpwindScheme, U, v)
    ṽ = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v_velocity(U))
    vᴿ = _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, bias(ṽ), v)
    regular_flux = ṽ * vᴿ
    polar_adjacent_row = (j == 1) | (j == grid.Ny)
    return ifelse(polar_adjacent_row, zero(grid), regular_flux)
end

@inline function projected_transport_div_𝐯u(i, j, k, grid::SphericalShellGrid, advection, U, u)
    scheme = projected_transport_momentum_scheme(grid, advection)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, _transport_momentum_flux_Uu, scheme, U, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, _transport_momentum_flux_Vu, scheme, U, u) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, scheme, w_velocity(U), u))
end

@inline function projected_transport_div_𝐯v(i, j, k, grid::SphericalShellGrid, advection, U, v)
    scheme = projected_transport_momentum_scheme(grid, advection)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _transport_momentum_flux_Uv, scheme, U, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, _transport_momentum_flux_Vv, scheme, U, v) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, scheme, w_velocity(U), v))
end

@inline _transport_flux_U_at_u(i, j, k, grid, scheme, U) =
    _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u_velocity(U))

@inline _transport_flux_V_at_u(i, j, k, grid, scheme, U) =
    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v_velocity(U))

@inline function _transport_flux_V_at_u(i, j, k, grid::OHPSG, scheme, U)
    regular_flux = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v_velocity(U))
    polar_fold = (j == 1) | (j == grid.Ny + 1)
    return ifelse(polar_fold, zero(grid), regular_flux)
end

@inline _transport_flux_U_at_v(i, j, k, grid, scheme, U) =
    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u_velocity(U))

@inline _transport_flux_V_at_v(i, j, k, grid, scheme, U) =
    _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v_velocity(U))

@inline function _transport_flux_V_at_v(i, j, k, grid::OHPSG, scheme, U)
    regular_flux = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v_velocity(U))
    polar_adjacent_row = (j == 1) | (j == grid.Ny)
    return ifelse(polar_adjacent_row, zero(grid), regular_flux)
end

@inline function projected_transport_divergence_at_u(i, j, k, grid::SphericalShellGrid, scheme, U)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, _transport_flux_U_at_u, scheme, U) +
                                    δyᵃᶜᵃ(i, j, k, grid, _transport_flux_V_at_u, scheme, U))
end

@inline function projected_transport_divergence_at_v(i, j, k, grid::SphericalShellGrid, scheme, U)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _transport_flux_U_at_v, scheme, U) +
                                    δyᵃᶠᵃ(i, j, k, grid, _transport_flux_V_at_v, scheme, U))
end

@inline function projected_transport_skew_div_𝐯u(i, j, k, grid::SphericalShellGrid, advection, U, u)
    scheme = projected_transport_momentum_scheme(grid, advection)
    conservative_advection = projected_transport_div_𝐯u(i, j, k, grid, advection, U, u)
    transport_divergence = projected_transport_divergence_at_u(i, j, k, grid, scheme, U)
    half = convert(eltype(grid), 1//2)
    @inbounds uᵢⱼₖ = u[i, j, k]
    return conservative_advection - half * uᵢⱼₖ * transport_divergence
end

@inline function projected_transport_skew_div_𝐯v(i, j, k, grid::SphericalShellGrid, advection, U, v)
    scheme = projected_transport_momentum_scheme(grid, advection)
    conservative_advection = projected_transport_div_𝐯v(i, j, k, grid, advection, U, v)
    transport_divergence = projected_transport_divergence_at_v(i, j, k, grid, scheme, U)
    half = convert(eltype(grid), 1//2)
    @inbounds vᵢⱼₖ = v[i, j, k]
    return conservative_advection - half * vᵢⱼₖ * transport_divergence
end

#####
##### Fallback advection fluxes!
#####

# Fallback for `nothing` advection
@inline _advective_momentum_flux_Uu(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Uv(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Uw(i, j, k, grid, ::Nothing, args...) = zero(grid)

@inline _advective_momentum_flux_Vu(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Vv(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Vw(i, j, k, grid, ::Nothing, args...) = zero(grid)

@inline _advective_momentum_flux_Wu(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Wv(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Ww(i, j, k, grid, ::Nothing, args...) = zero(grid)
