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

@inline function advective_momentum_flux_Uu(i, j, k, grid, scheme::UpwindScheme, U, u, is, js, ks)

    ũ  =    _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, U, is, js, ks)
    uᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u, is, js, ks)
    uᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u, is, js, ks)

    return Axᶜᶜᶜ(i, j, k, grid) * upwind_biased_product(ũ, uᴸ, uᴿ)
end

@inline function advective_momentum_flux_Vu(i, j, k, grid, scheme::UpwindScheme, V, u)

    ṽ  =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, V, is, js, ks)
    uᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u, is, js, ks)
    uᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u, is, js, ks)

    return Ayᶠᶠᶜ(i, j, k, grid) * upwind_biased_product(ṽ, uᴸ, uᴿ)
end

@inline function advective_momentum_flux_Wu(i, j, k, grid, scheme::UpwindScheme, W, u)

    w̃  =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W, is, js, ks)
    uᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u, is, js, ks)
    uᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u, is, js, ks)

    return Azᶠᶜᶠ(i, j, k, grid) * upwind_biased_product(w̃, uᴸ, uᴿ)
end

@inline function advective_momentum_flux_Uv(i, j, k, grid, scheme::UpwindScheme, U, v)

    ũ  =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, U, is, js, ks)
    vᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v, is, js, ks)
    vᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v, is, js, ks)
 
    return Axᶠᶠᶜ(i, j, k, grid) * upwind_biased_product(ũ, vᴸ, vᴿ)
end

@inline function advective_momentum_flux_Vv(i, j, k, grid, scheme::UpwindScheme, V, v)

    ṽ  =    _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, V, is, js, ks)
    vᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v, is, js, ks)
    vᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v, is, js, ks)

    return Ayᶜᶜᶜ(i, j, k, grid) * upwind_biased_product(ṽ, vᴸ, vᴿ)
end

@inline function advective_momentum_flux_Wv(i, j, k, grid, scheme::UpwindScheme, W, v)

    w̃  =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W, is, js, ks)
    vᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v, is, js, ks)
    vᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v, is, js, ks)

    return Azᶜᶠᶠ(i, j, k, grid) * upwind_biased_product(w̃, vᴸ, vᴿ)
end

@inline function advective_momentum_flux_Uw(i, j, k, grid, scheme::UpwindScheme, U, w)

    ũ  =    _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, U, is, js, ks)
    wᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w, is, js, ks)
    wᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w, is, js, ks)

    return Axᶠᶜᶠ(i, j, k, grid) * upwind_biased_product(ũ, wᴸ, wᴿ)
end

@inline function advective_momentum_flux_Vw(i, j, k, grid, scheme::UpwindScheme, V, w)

    ṽ  =    _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, V, is, js, ks)
    wᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w, is, js, ks)
    wᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w, is, js, ks)

    return Ayᶜᶠᶠ(i, j, k, grid) * upwind_biased_product(ṽ, wᴸ, wᴿ)
end

@inline function advective_momentum_flux_Ww(i, j, k, grid, scheme::UpwindScheme, W, w)

    w̃  =    _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, W, is, js, ks)
    wᴸ =  _left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w, is, js, ks)
    wᴿ = _right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w, is, js, ks)

    return Azᶜᶜᶜ(i, j, k, grid) * upwind_biased_product(w̃, wᴸ, wᴿ)
end

#####
##### Tracer advection operators
#####
    
@inline function advective_tracer_flux_x(i, j, k, grid, scheme::UpwindScheme, U, c, is, js, ks)

    @inbounds ũ = U[is, js, ks]
    cᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c, is, js, ks)
    cᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c, is, js, ks)

    return Axᶠᶜᶜ(i, j, k, grid) * upwind_biased_product(ũ, cᴸ, cᴿ)
end

@inline function advective_tracer_flux_y(i, j, k, grid, scheme::UpwindScheme, V, c, is, js, ks)

    @inbounds ṽ = V[is, js, ks]
    cᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c, is, js, ks)
    cᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c, is, js, ks)

    return Ayᶜᶠᶜ(i, j, k, grid) * upwind_biased_product(ṽ, cᴸ, cᴿ)
end

@inline function advective_tracer_flux_z(i, j, k, grid, scheme::UpwindScheme, W, c, is, js, ks)

    @inbounds w̃ = W[is, js, ks]
    cᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)
    cᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)

    return Azᶜᶜᶠ(i, j, k, grid) * upwind_biased_product(w̃, cᴸ, cᴿ, is, js, ks) 
end
