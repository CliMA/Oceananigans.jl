const VectorInvariantCrossVerticalUpwinding = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:CrossAndSelfUpwinding}

#####
##### Cross upwinding results in the largest kinetic energy content,
##### but because of presence of mixed upwinding leading to cross-double derivatives
##### it is slightly unstable at larger orders. 
#####

##### 
##### Due to the presence of cross derivative terms that generate excessive noise and result in 
##### numerical instabilities, it is not possible to perform a complete upwinding of the Kinetic 
##### Energy gradient. Consequently, the `OnlySelfUpwinding` scheme is implemented for the Kinetic 
##### Energy gradient in the case of `CrossAndSelfUpwinding`.
##### For details on the implementation refer to the file `vector_invariant_self_upwinding.jl` 
#####

#####
##### Cross and Self Upwinding of the Divergence flux
#####

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantCrossVerticalUpwinding, u, v)
    @inbounds û = u[i, j, k]
    δ_stencil = scheme.upwinding_treatment.divergence_stencil

    side = upwinding_direction(û)
    δ    =  _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, side, flux_div_xyᶜᶜᶜ, δ_stencil, u, v) 

    return û * δ
end

@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantCrossVerticalUpwinding, u, v)
    @inbounds v̂ = v[i, j, k]
    δ_stencil = scheme.upwinding_treatment.divergence_stencil

    side = upwinding_direction(v̂)
    δ    =  _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, side, flux_div_xyᶜᶜᶜ, δ_stencil, u, v) 

    return v̂ * δ
end
