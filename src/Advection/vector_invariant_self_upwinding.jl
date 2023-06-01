const VectorInvariantSelfVerticalUpwinding = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:SelfUpwinding}

##### 
##### Self Upwinding of Divergence Flux, the best option!
#####

@inline δx_U(i, j, k, grid, u, v) =  δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline δy_V(i, j, k, grid, u, v) =  δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# Velocity smoothness for divergence upwinding
@inline U_smoothness(i, j, k, grid, u, v) = ℑxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline V_smoothness(i, j, k, grid, u, v) = ℑyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# Divergence smoothness for divergence upwinding
@inline divergence_smoothness(i, j, k, grid, u, v) = δx_U(i, j, k, grid, u, v) + δy_V(i, j, k, grid, u, v)

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, u, v)
    @inbounds û = u[i, j, k]
    δvˢ =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δy_V, u, v) 
    δuᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δx_U, scheme.δU_stencil, u, v) 
    δuᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δx_U, scheme.δU_stencil, u, v) 

    return upwind_biased_product(û, δuᴸ, δuᴿ) + û * δvˢ
end

@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, u, v)
    @inbounds v̂ = v[i, j, k]
    δuˢ =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δx_U, u, v) 
    δvᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δy_V, scheme.δV_stencil, u, v) 
    δvᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δy_V, scheme.δV_stencil, u, v) 

    return upwind_biased_product(v̂, δvᴸ, δvᴿ) + v̂ * δuˢ
end

#####
##### Self Upwinding of Kinetic Energy Gradient 
#####

const VectorInvariantVerticalUpwinding = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme}

@inline half_ϕ²(i, j, k, grid, ϕ) = ϕ[i, j, k]^2 / 2

@inline δx_u²(i, j, k, grid, u, v) = δxᶜᶜᶜ(i, j, k, grid, half_ϕ², u)
@inline δy_u²(i, j, k, grid, u, v) = δyᶜᶠᶜ(i, j, k, grid, half_ϕ², u)

@inline δx_v²(i, j, k, grid, u, v) = δxᶠᶜᶜ(i, j, k, grid, half_ϕ², v)
@inline δy_v²(i, j, k, grid, u, v) = δyᶜᶜᶜ(i, j, k, grid, half_ϕ², v)

@inline u_smoothness(i, j, k, grid, u, v) = ℑxᶜᶜᶜ(i, j, k, grid, u)
@inline v_smoothness(i, j, k, grid, u, v) = ℑyᶜᶜᶜ(i, j, k, grid, v)

@inline function bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantVerticalUpwinding, u, v)

    @inbounds û = u[i, j, k]

    δKvˢ =    _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δx_v², u, v)
    δKuᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δx_u², scheme.δu²_stencil, u, v)
    δKuᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δx_u², scheme.δu²_stencil, u, v)
    
    ∂Kᴸ = (δKuᴸ + δKvˢ) / Δxᶠᶜᶜ(i, j, k, grid)
    ∂Kᴿ = (δKuᴿ + δKvˢ) / Δxᶠᶜᶜ(i, j, k, grid)

    return ifelse(û > 0, ∂Kᴸ, ∂Kᴿ)
end

@inline function bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantVerticalUpwinding, u, v)

    @inbounds v̂ = v[i, j, k]

    δKuˢ =    _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δy_u², u, v)
    δKvᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δy_v², scheme.δv²_stencil, u, v) 
    δKvᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δy_v², scheme.δv²_stencil, u, v) 
    
    ∂Kᴸ = (δKvᴸ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid) 
    ∂Kᴿ = (δKvᴿ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid)

    return ifelse(v̂ > 0, ∂Kᴸ, ∂Kᴿ)
end