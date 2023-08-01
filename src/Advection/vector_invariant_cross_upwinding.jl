const VectorInvariantCrossVerticalUpwinding = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:CrossAndSelfUpwinding}

#####
##### Cross upwinding results in the largest kinetic energy content, 
##### but because of presence of mixed upwinding the truncation error of 
##### the numerical discretization is not always negative definite at 
##### leading (diffusive) order. This scheme might be unstable at larger orders. 
#####

##### 
##### Due to the presence of cross derivative terms that generate excessive noise and result in 
##### numerical instabilities, it is undesirable to perform a complete upwinding of the Kinetic 
##### Energy gradient. Consequently, the `OnlySelfUpwinding` scheme is implemented for the Kinetic 
##### Energy gradient in the case of `CrossAndSelfUpwinding`.
##### For details on the implementation refer to the file `vector_invariant_self_upwinding.jl` 
#####

#####
##### Cross and Self Upwinding of the Divergence flux
#####

@inline function upwinded_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantCrossVerticalUpwinding, u, v)
    @inbounds û = u[i, j, k]
    δ_stencil = scheme.upwinding.divergence_stencil

    δᴿ = upwind_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, û, scheme, scheme.vertical_scheme, flux_div_xyᶜᶜᶜ, δ_stencil, u, v) 

    return û * δᴿ
end

@inline function upwinded_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantCrossVerticalUpwinding, u, v)
    @inbounds v̂ = v[i, j, k]
    δ_stencil = scheme.upwinding.divergence_stencil

    δᴿ = upwind_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, v̂, scheme, scheme.vertical_scheme, flux_div_xyᶜᶜᶜ, δ_stencil, u, v) 

    return v̂ * δᴿ
end
