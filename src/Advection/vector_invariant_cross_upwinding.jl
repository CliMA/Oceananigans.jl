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

@inline V_times_∂t_e₃(i, j, k, grid) = Vᶜᶜᶜ(i, j, k, grid) * ∂t_e₃(i, j, k, grid)

@inline function upwinded_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantCrossVerticalUpwinding, u, v)
    @inbounds û = u[i, j, k]
    δ_stencil = scheme.upwinding.divergence_stencil

    δᴿ  =    _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, bias(û), flux_div_xyᶜᶜᶜ, δ_stencil, u, v) 
    ∂ts = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, cross_scheme, V_times_∂t_e₃)

    return û * (δᴿ + ∂ts)
end

@inline function upwinded_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantCrossVerticalUpwinding, u, v)
    @inbounds v̂ = v[i, j, k]
    δ_stencil = scheme.upwinding.divergence_stencil

    δᴿ  =    _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, bias(v̂), flux_div_xyᶜᶜᶜ, δ_stencil, u, v) 
    ∂ts = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, cross_scheme, V_times_∂t_e₃)

    return v̂ * (δᴿ + ∂ts)
end
