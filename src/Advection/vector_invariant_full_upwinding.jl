const VectorInvariantFullVerticalUpwinding = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:FullUpwinding}

#####
##### Full upwindind results in the largest energy content and lowest
##### spurious mixing, but because of resence of mixed upwinding leading to cross-double derivatives
##### it is slightly unstable at larger orders. 
#####

##### 
##### Due to the presence of cross derivative terms that generate excessive noise and result in 
##### numerical instabilities, it is not possible to perform a complete upwinding of the Kinetic 
##### Energy gradient. Consequently, a `PartialUpwinding` scheme is implemented for the Kinetic 
##### Energy gradient in the case of `FullUpwinding`. Please refer to the file 
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
