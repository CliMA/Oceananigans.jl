#####
##### Momentum and tracer advective flux operators for upwind-biased advection schemes
#####

const UpwindScheme = AbstractUpwindBiasedAdvectionScheme
const ETD = ExplicitTimeDiscretization

@inline upwind_biased_product(ũ, ψᴸ, ψᴿ) = ((ũ + abs(ũ)) * ψᴸ + (ũ - abs(ũ)) * ψᴿ) / 2

#####
##### Momentum advection operators
#####
##### Note the convention "advective_momentum_flux_AB" corresponds to the advection _of_ B _by_ A.
#####

@enum Bias LeftBias RightBias

@inline bias(u::Number) = ifelse(u > 0, LeftBias, RightBias)

@inline function advective_momentum_flux_Uu(i, j, k, grid, scheme::UpwindScheme, ::ETD, U, u)

    ũ  = ℑxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, U)
    uᴿ = biased_interpolate_xᶜᶜᶜ(i, j, k, grid, scheme, bias(ũ), u)

    return ũ * uᴿ
end

@inline function advective_momentum_flux_Vu(i, j, k, grid, scheme::UpwindScheme, ::ETD, V, u)

    ṽ  = ℑxᶠᵃᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, V)
    uᴿ = biased_interpolate_yᶠᶠᶜ(i, j, k, grid, scheme, bias(ṽ), u)

    return ṽ * uᴿ
end

@inline function advective_momentum_flux_Wu(i, j, k, grid, scheme::UpwindScheme, ::ETD, W, u)

    w̃  = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, W)
    uᴿ = biased_interpolate_zᶠᶜᶠ(i, j, k, grid, scheme, bias(w̃), u)

    return w̃ * uᴿ
end

@inline function advective_momentum_flux_Uv(i, j, k, grid, scheme::UpwindScheme, ::ETD, U, v)

    ũ  = ℑyᵃᶠᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, U)
    vᴿ = biased_interpolate_xᶠᶠᶜ(i, j, k, grid, scheme, bias(ũ), v)

    return ũ * vᴿ
end

@inline function advective_momentum_flux_Vv(i, j, k, grid, scheme::UpwindScheme, ::ETD, V, v)

    ṽ  = ℑyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, V)
    vᴿ = biased_interpolate_yᶜᶜᶜ(i, j, k, grid, scheme, bias(ṽ), v)

    return ṽ * vᴿ
end

@inline function advective_momentum_flux_Wv(i, j, k, grid, scheme::UpwindScheme, ::ETD, W, v)

    w̃  = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, W)
    vᴿ = biased_interpolate_zᶜᶠᶠ(i, j, k, grid, scheme, bias(w̃), v)

    return w̃ * vᴿ
end

@inline function advective_momentum_flux_Uw(i, j, k, grid, scheme::UpwindScheme, ::ETD, U, w)

    ũ  = ℑzᵃᵃᶠ(i, j, k, grid, Ax_qᶠᶜᶜ, U)
    wᴿ = biased_interpolate_xᶠᶜᶠ(i, j, k, grid, scheme, bias(ũ), w)

    return ũ * wᴿ
end

@inline function advective_momentum_flux_Vw(i, j, k, grid, scheme::UpwindScheme, ::ETD, V, w)

    ṽ  = ℑzᵃᵃᶠ(i, j, k, grid, Ay_qᶜᶠᶜ, V)
    wᴿ = biased_interpolate_yᶜᶠᶠ(i, j, k, grid, scheme, bias(ṽ), w)

    return ṽ * wᴿ
end

@inline function advective_momentum_flux_Ww(i, j, k, grid, scheme::UpwindScheme, ::ETD, W, w)

    w̃  = ℑzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, W)
    wᴿ = biased_interpolate_zᶜᶜᶜ(i, j, k, grid, scheme, bias(w̃), w)

    return w̃ * wᴿ
end

#####
##### Tracer advection operators
#####

@inline function advective_tracer_flux_x(i, j, k, grid, scheme::UpwindScheme, ::ETD, U, c)

    @inbounds ũ = U[i, j, k]
    cᴿ = tracer_interpolate(i, j, k, grid, Val(1), (Face(), Center(), Center()), scheme, bias(ũ), c)

    return Axᶠᶜᶜ(i, j, k, grid) * ũ * cᴿ
end

@inline function advective_tracer_flux_y(i, j, k, grid, scheme::UpwindScheme, ::ETD, V, c)

    @inbounds ṽ = V[i, j, k]
    cᴿ = tracer_interpolate(i, j, k, grid, Val(2), (Center(), Face(), Center()), scheme, bias(ṽ), c)

    return Ayᶜᶠᶜ(i, j, k, grid) * ṽ * cᴿ
end

@inline function advective_tracer_flux_z(i, j, k, grid, scheme::UpwindScheme, ::ETD, W, c)

    @inbounds w̃ = W[i, j, k]
    cᴿ = tracer_interpolate(i, j, k, grid, Val(3), (Center(), Center(), Face()), scheme, bias(w̃), c)

    return Azᶜᶜᶠ(i, j, k, grid) * w̃ * cᴿ
end
