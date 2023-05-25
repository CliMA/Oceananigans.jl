const VectorInvariantPartialVerticalUpwinding = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:PartialUpwinding}

##### 
##### Partial Upwinding of Divergence Flux (Most stable formulation)
#####

@inline δ_U(i, j, k, grid, u, v) =  δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline δ_V(i, j, k, grid, u, v) =  δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# Velocity smoothness for divergence upwinding
@inline velocity_smoothness_U(i, j, k, grid, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline velocity_smoothness_V(i, j, k, grid, u, v) = ℑyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# Divergence smoothness for divergence upwinding
@inline divergence_smoothness(i, j, k, grid, u, v) = δ_U(i, j, k, grid, u, v) + δ_V(i, j, k, grid, u, v)

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantPartialVerticalUpwinding, u, v)
    @inbounds û = u[i, j, k]
    δvˢ =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.vertical_scheme, δ_V, u, v) 
    δuᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.vertical_scheme, δ_U, scheme.u_stencil, u, v) 
    δuᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.vertical_scheme, δ_U, scheme.u_stencil, u, v) 

    return upwind_biased_product(û, δuᴸ, δuᴿ) + û * δvˢ
end

@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantPartialVerticalUpwinding, u, v)
    @inbounds v̂ = v[i, j, k]
    δuˢ =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.vertical_scheme, δ_U, u, v) 
    δvᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.vertical_scheme, δ_V, scheme.v_stencil, u, v) 
    δvᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.vertical_scheme, δ_V, scheme.v_stencil, u, v) 

    return upwind_biased_product(v̂, δvᴸ, δvᴿ) + v̂ * δuˢ
end