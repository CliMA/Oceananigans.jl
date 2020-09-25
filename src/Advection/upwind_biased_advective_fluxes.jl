@inline upwind_biased_product(ũ, ψᴸ, ψᴿ) = ((ũ + abs(ũ)) * ψᴸ + (ũ - abs(ũ)) * ψᴿ) / 2

""" Advection of u by u. """
@inline function momentum_flux_uu(i, j, k, grid, scheme, u)

    ũ  =    _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u) 
    uᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)
    uᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)

    return Axᵃᵃᶠ(i, j, k, grid) * upwind_biased_product(ũ, uᴸ, uᴿ)
end

""" Advection of u by v. """
@inline function momentum_flux_uv(i, j, k, grid, scheme, u, v)

    ṽ  =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
    uᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
    uᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)

    return Ayᵃᵃᶠ(i, j, k, grid) * upwind_biased_product(ṽ, uᴸ, uᴿ)
end

""" Advection of u by w. """
@inline function momentum_flux_uw(i, j, k, grid, scheme, u, w)

    w̃  =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)
    uᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u)
    uᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u)

    return Azᵃᵃᵃ(i, j, k, grid) * upwind_biased_product(w̃, uᴸ, uᴿ)
end

""" Advection of v by u. """
@inline function momentum_flux_vu(i, j, k, grid, scheme, u, v)

    ũ  =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
    vᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
    vᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
 
    return Axᵃᵃᶠ(i, j, k, grid) * upwind_biased_product(ũ, vᴸ, vᴿ)
end

""" Advection of v by v. """
@inline function momentum_flux_vv(i, j, k, grid, scheme, v)

    ṽ  =    _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
    vᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
    vᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)

    return Ayᵃᵃᶠ(i, j, k, grid) * upwind_biased_product(ṽ, vᴸ, vᴿ)
end

""" Advection of v by w. """
@inline function momentum_flux_vw(i, j, k, grid, scheme, u, w)

    w̃  =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)
    vᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v)
    vᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v)

    return Azᵃᵃᵃ(i, j, k, grid) * upwind_biased_product(w̃, vᴸ, vᴿ)
end

""" Advection of u by w. """
@inline function momentum_flux_wu(i, j, k, grid, scheme, u, w)

    ũ  =    _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u)
    wᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)
    wᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)

    return Axᵃᵃᶠ(i, j, k, grid) * upwind_biased_product(ũ, wᴸ, wᴿ)
end

@inline function momentum_flux_wv(i, j, k, grid, scheme, v, w)

    ṽ  =    _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v)
    wᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)
    wᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)

    return Ayᵃᵃᶠ(i, j, k, grid) * upwind_biased_product(ṽ, wᴸ, wᴿ)
end

@inline function momentum_flux_ww(i, j, k, grid, scheme, w)

    w̃  =    _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)
    wᴸ =  _left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)
    wᴿ = _right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)

    return Azᵃᵃᵃ(i, j, k, grid) * upwind_biased_product(w̃, wᴸ, wᴿ)
    
@inline function advective_tracer_flux_x(i, j, k, grid, scheme, u, c) 

    @inbounds ũ = u[i, j, k]
    cᴸ = _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
    cᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)

    Axᵃᵃᶠ(i, j, k, grid) * upwind_biased_product(ũ, cᴸ, cᴿ)
end

@inline function advective_tracer_flux_y(i, j, k, grid, scheme, v, c)

    @inbounds ṽ = v[i, j, k]
    cᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
    cᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)

    return Ayᵃᵃᶠ(i, j, k, grid) * upwind_biased_product(ṽ, cᴸ, cᴿ)
end

@inline function advective_tracer_flux_z(i, j, k, grid, scheme, w, c)

    @inbounds w̃ = w[i, j, k]
    cᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)
    cᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)

    return Az_ψᵃᵃᵃ(i, j, k, grid) * upwind_biased_product(w̃, cᴸ, cᴿ) 
end
