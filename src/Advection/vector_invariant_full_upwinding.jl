const VectorInvariantFullVerticalUpwinding = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:FullUpwinding}

#####
##### Full upwindind results in the largest energy content and lowest
##### spurious mixing, but is slightly unstable at larger orders
#####

#####
##### Full Upwinding of the Divergence flux
#####

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantFullVerticalUpwinding, u, v)
    @inbounds û = u[i, j, k]
    δᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, flux_div_xyᶜᶜᶜ, u, v) 
    δᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, flux_div_xyᶜᶜᶜ, u, v) 

    return upwind_biased_product(û, δᴸ, δᴿ)
end

@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantFullVerticalUpwinding, u, v)
    @inbounds v̂ = v[i, j, k]
    δᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, flux_div_xyᶜᶜᶜ, u, v) 
    δᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, flux_div_xyᶜᶜᶜ, u, v) 

    return upwind_biased_product(v̂, δᴸ, δᴿ) 
end

##### 
##### Full Upwinding entails a Partial Upwind of the Bernoulli term because
##### cross derivative terms caused by cross-upwinding generate excessive noise
##### that result in instabilities
#####

@inline half_ϕ²(i, j, k, grid, ϕ) = ϕ[i, j, k]^2 / 2

@inline function bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantFullVerticalUpwinding, u, v)

    @inbounds û = u[i, j, k]

    δKvˢ =    _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δxᶠᵃᵃ, half_ϕ², v) 
    δKuᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δxᶜᵃᵃ, half_ϕ², u)
    δKuᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δxᶜᵃᵃ, half_ϕ², u)
    
    ∂Kᴸ = (δKuᴸ + δKvˢ) / Δxᶠᶜᶜ(i, j, k, grid)
    ∂Kᴿ = (δKuᴿ + δKvˢ) / Δxᶠᶜᶜ(i, j, k, grid)

    return ifelse(û > 0, ∂Kᴸ, ∂Kᴿ)
end

@inline function bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantFullVerticalUpwinding, u, v)

    @inbounds v̂ = v[i, j, k]

    δKuˢ =    _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δyᵃᶠᵃ, half_ϕ², u)
    δKvᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δyᵃᶜᵃ, half_ϕ², v) 
    δKvᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δyᵃᶜᵃ, half_ϕ², v) 
    
    ∂Kᴸ = (δKvᴸ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid) 
    ∂Kᴿ = (δKvᴿ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid)

    return ifelse(v̂ > 0, ∂Kᴸ, ∂Kᴿ)
end