using Oceananigans.Grids: AbstractPrimaryGrid

#####
##### Momentum and tracer advective flux operators for upwind-biased advection schemes
#####
##### See topologically_conditional_interpolation.jl for an explanation of the underscore-prepended
##### functions _symmetric_interpolate_*, _left_biased_interpolate_*, and _right_biased_interpolate_*.
#####

const Upwind = AbstractUpwindBiasedAdvectionScheme

@inline upwind_biased_product(ũ, ψᴸ, ψᴿ) = ((ũ + abs(ũ)) * ψᴸ + (ũ - abs(ũ)) * ψᴿ) / 2

#####
##### Momentum advection operators
#####
##### Note the convention "advective_momentum_flux_AB" corresponds to the advection _of_ B _by_ A.
#####

@inline function advective_momentum_flux_Uu(i, j, k, grid, scheme::Upwind, U, u)

    ũ  =    _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, U)
    uᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)
    uᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, u)

    return Axᶜᶜᶜ(i, j, k, grid) * upwind_biased_product(ũ, uᴸ, uᴿ)
end

@inline function advective_momentum_flux_Vu(i, j, k, grid, scheme::Upwind, V, u)

    ṽ  =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, V)
    uᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)
    uᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, u)

    return Ayᶠᶠᶜ(i, j, k, grid) * upwind_biased_product(ṽ, uᴸ, uᴿ)
end

@inline function advective_momentum_flux_Wu(i, j, k, grid, scheme::Upwind, W, u)

    w̃  =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W)
    uᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u)
    uᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, u)

    return Azᶠᶜᵃ(i, j, k, grid) * upwind_biased_product(w̃, uᴸ, uᴿ)
end

@inline function advective_momentum_flux_Uv(i, j, k, grid, scheme::Upwind, U, v)

    ũ  =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, U)
    vᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
    vᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, v)
 
    return Axᶠᶠᶜ(i, j, k, grid) * upwind_biased_product(ũ, vᴸ, vᴿ)
end

@inline function advective_momentum_flux_Vv(i, j, k, grid, scheme::Upwind, V, v)

    ṽ  =    _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, V)
    vᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)
    vᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, v)

    return Ayᶜᶜᶜ(i, j, k, grid) * upwind_biased_product(ṽ, vᴸ, vᴿ)
end

@inline function advective_momentum_flux_Wv(i, j, k, grid, scheme::Upwind, W, v)

    w̃  =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W)
    vᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v)
    vᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, v)

    return Azᶜᶠᵃ(i, j, k, grid) * upwind_biased_product(w̃, vᴸ, vᴿ)
end

@inline function advective_momentum_flux_Uw(i, j, k, grid, scheme::Upwind, U, w)

    ũ  =    _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, U)
    wᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)
    wᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, w)

    return Axᶠᶜᶠ(i, j, k, grid) * upwind_biased_product(ũ, wᴸ, wᴿ)
end

@inline function advective_momentum_flux_Vw(i, j, k, grid, scheme::Upwind, V, w)

    ṽ  =    _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, V)
    wᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)
    wᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, w)

    return Ayᵃᵃᶠ(i, j, k, grid) * upwind_biased_product(ṽ, wᴸ, wᴿ)
end

@inline function advective_momentum_flux_Ww(i, j, k, grid, scheme::Upwind, W, w)

    w̃  =    _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, W)
    wᴸ =  _left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)
    wᴿ = _right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, w)

    return Azᶜᶠᵃ(i, j, k, grid) * upwind_biased_product(w̃, wᴸ, wᴿ)
end

#####
##### Tracer advection operators
#####
    
@inline function advective_tracer_flux_x(i, j, k, grid, scheme::Upwind, U, c) 

    @inbounds ũ = U[i, j, k]
    cᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
    cᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)

    return Axᶠᶜᶜ(i, j, k, grid) * upwind_biased_product(ũ, cᴸ, cᴿ)
end

@inline function advective_tracer_flux_y(i, j, k, grid, scheme::Upwind, V, c)

    @inbounds ṽ = V[i, j, k]
    cᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
    cᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)

    return Ayᶜᶠᶜ(i, j, k, grid) * upwind_biased_product(ṽ, cᴸ, cᴿ)
end

@inline function advective_tracer_flux_z(i, j, k, grid, scheme::Upwind, W, c)

    @inbounds w̃ = W[i, j, k]
    cᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)
    cᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)

    return Azᶜᶜᵃ(i, j, k, grid) * upwind_biased_product(w̃, cᴸ, cᴿ) 
end
