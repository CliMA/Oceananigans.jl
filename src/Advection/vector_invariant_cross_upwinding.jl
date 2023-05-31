const VectorInvariantCrossVerticalUpwinding = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:CrossUpwinding}

#####
##### Cross upwinding results in the largest kinetic energy content,
##### but because of presence of mixed upwinding leading to cross-double derivatives
##### it is slightly unstable at larger orders. 
#####

##### 
##### Due to the presence of cross derivative terms that generate excessive noise and result in 
##### numerical instabilities, it is not possible to perform a complete upwinding of the Kinetic 
##### Energy gradient. Consequently, a `SelfUpwinding` scheme is implemented for the Kinetic 
##### Energy gradient in the case of `CrossUpwinding`. Please refer to the file 
#####

#####
##### Cross Upwinding of the Divergence flux
#####

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantCrossVerticalUpwinding, u, v)
    @inbounds û = u[i, j, k]
    δᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, flux_div_xyᶜᶜᶜ, u, v) 
    δᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, flux_div_xyᶜᶜᶜ, u, v) 

    return upwind_biased_product(û, δᴸ, δᴿ)
end

@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantCrossVerticalUpwinding, u, v)
    @inbounds v̂ = v[i, j, k]
    δᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, flux_div_xyᶜᶜᶜ, u, v) 
    δᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, flux_div_xyᶜᶜᶜ, u, v) 

    return upwind_biased_product(v̂, δᴸ, δᴿ) 
end
