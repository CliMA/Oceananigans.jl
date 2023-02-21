#####
##### Momentum and tracer advective flux operators for upwind-biased advection schemes
#####
##### See topologically_conditional_interpolation.jl for an explanation of the underscore-prepended
##### functions _symmetric_interpolate_* and _biased_interpolate_*
#####

const UpwindScheme = AbstractUpwindBiasedAdvectionScheme

@inline upwinding_direction(ũ) = ifelse(ũ > 0, Val(:left), Val(:right))

#####
##### Momentum advection operators
#####
##### Note the convention "advective_momentum_flux_AB" corresponds to the advection _of_ B _by_ A.
#####

@inline function advective_momentum_flux_Uu(i, j, k, grid, scheme::UpwindScheme, U, u)

    ũ    = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, U)
    side = upwinding_direction(ũ) 
    uᴿ   =  _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, side, u)

    return Axᶜᶜᶜ(i, j, k, grid) * ũ * uᴿ
end

@inline function advective_momentum_flux_Vu(i, j, k, grid, scheme::UpwindScheme, V, u)

    ṽ    = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, V)
    side = upwinding_direction(ṽ) 
    uᴿ   = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, side, u)

    return Ayᶠᶠᶜ(i, j, k, grid) * ṽ * uᴿ
end

@inline function advective_momentum_flux_Wu(i, j, k, grid, scheme::UpwindScheme, W, u)

    w̃    = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, W)
    side = upwinding_direction(w̃) 
    uᴿ   =  _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, side, u)

    return Azᶠᶜᶠ(i, j, k, grid) * w̃ * uᴿ
end

@inline function advective_momentum_flux_Uv(i, j, k, grid, scheme::UpwindScheme, U, v)

    ũ    = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, U)
    side = upwinding_direction(ũ) 
    vᴿ   = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, side, v)
 
    return Axᶠᶠᶜ(i, j, k, grid) * ũ * vᴿ
end

@inline function advective_momentum_flux_Vv(i, j, k, grid, scheme::UpwindScheme, V, v)

    ṽ    = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, V)
    side = upwinding_direction(ṽ) 
    vᴿ   = _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, side, v)

    return Ayᶜᶜᶜ(i, j, k, grid) * ṽ * vᴿ
end

@inline function advective_momentum_flux_Wv(i, j, k, grid, scheme::UpwindScheme, W, v)

    w̃    = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W)
    side = upwinding_direction(w̃) 
    vᴿ   = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, side, v)

    return Azᶜᶠᶠ(i, j, k, grid) * w̃ * vᴿ
end

@inline function advective_momentum_flux_Uw(i, j, k, grid, scheme::UpwindScheme, U, w)

    ũ   = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, U)
    side = upwinding_direction(ũ) 
    wᴿ   = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, side, w)

    return Axᶠᶜᶠ(i, j, k, grid) * ũ * wᴿ
end

@inline function advective_momentum_flux_Vw(i, j, k, grid, scheme::UpwindScheme, V, w)

    ṽ    = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, V)
    side = upwinding_direction(ṽ)
    wᴿ   = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, side, w)

    return Ayᶜᶠᶠ(i, j, k, grid) * ṽ * wᴿ
end

@inline function advective_momentum_flux_Ww(i, j, k, grid, scheme::UpwindScheme, W, w)

    w̃    = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, W)
    side = upwinding_direction(w̃) 
    wᴿ   = _biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, side, w)

    return Azᶜᶜᶜ(i, j, k, grid) * w̃ * wᴿ
end

#####
##### Tracer advection operators
#####
    
@inline function advective_tracer_flux_x(i, j, k, grid, scheme::UpwindScheme, U, c) 

    @inbounds ũ = U[i, j, k]
    side = upwinding_direction(ũ) 
    cᴿ   = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, side, c)

    return Axᶠᶜᶜ(i, j, k, grid) * ũ * cᴿ
end

@inline function advective_tracer_flux_y(i, j, k, grid, scheme::UpwindScheme, V, c)

    @inbounds ṽ = V[i, j, k]
    side = upwinding_direction(ṽ) 
    cᴿ   = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, side, c)

    return Ayᶜᶠᶜ(i, j, k, grid) * ṽ * cᴿ
end

@inline function advective_tracer_flux_z(i, j, k, grid, scheme::UpwindScheme, W, c)

    @inbounds w̃ = W[i, j, k]
    side = upwinding_direction(w̃) 
    cᴿ   = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, side, c)

    return Azᶜᶜᶠ(i, j, k, grid) * w̃ * cᴿ
end
