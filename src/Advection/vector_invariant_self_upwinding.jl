##### 
##### Self Upwinding of Divergence Flux, the best option!
#####

# Velocity smoothness for divergence upwinding (not used, it leads to very large dissipation)
@inline U_smoothness(i, j, k, grid, δ, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline V_smoothness(i, j, k, grid, δ, u, v) = ℑyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# Divergence smoothness for divergence upwinding (if δ is precomputed use δ)
@inline divergence_smoothness(i, j, k, grid, ::Nothing, u, v) = flux_div_xyᶜᶜᶜ(i, j, k, grid, u, v) 
@inline divergence_smoothness(i, j, k, grid, δ, args...)      = @inbounds δ[i, j, k]

@inline function upwinded_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, u, v)

    δU_stencil        = scheme.upwinding.δU_stencil    
    cross_scheme      = scheme.upwinding.cross_scheme
    divergence_scheme = scheme.divergence_scheme

    δx_U = scheme.auxiliary_fields.δx_U
    δy_V = scheme.auxiliary_fields.δy_V
    δ    = scheme.auxiliary_fields.δ

    @inbounds û = u[i, j, k]
    δvˢ =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, cross_scheme, δy_V, δ, u, v) 
    δuᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, divergence_scheme, δx_U, δU_stencil, δ, u, v) 
    δuᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, divergence_scheme, δx_U, δU_stencil, δ, u, v) 

    return upwind_biased_product(û, δuᴸ, δuᴿ) + û * δvˢ
end

@inline function upwinded_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, u, v)
    
    δV_stencil        = scheme.upwinding.δV_stencil
    cross_scheme      = scheme.upwinding.cross_scheme
    divergence_scheme = scheme.divergence_scheme

    δx_U = scheme.auxiliary_fields.δx_U
    δy_V = scheme.auxiliary_fields.δy_V
    δ    = scheme.auxiliary_fields.δ

    @inbounds v̂ = v[i, j, k]
    δuˢ =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, cross_scheme, δx_U, δ, u, v)
    δvᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, divergence_scheme, δy_V, δV_stencil, δ, u, v) 
    δvᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, divergence_scheme, δy_V, δV_stencil, δ, u, v) 

    return upwind_biased_product(v̂, δvᴸ, δvᴿ) + v̂ * δuˢ
end

#####
##### Self Upwinding of Kinetic Energy Gradient 
#####

@inline u_smoothness(i, j, k, grid, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline v_smoothness(i, j, k, grid, u, v) = ℑyᵃᶜᵃ(i, j, k, grid, v)

@inline function bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantKineticEnergyUpwinding, u, v)

    @inbounds û = u[i, j, k]

    δu²_stencil  = scheme.upwinding.δu²_stencil    
    cross_scheme = scheme.upwinding.cross_scheme
    δx_u²        = scheme.auxiliary_fields.δx_u²
    
    δKvˢ =    _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, cross_scheme, δx_v², u, v)
    δKuᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.kinetic_energy_gradient_scheme, δx_u², δu²_stencil, u, v)
    δKuᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.kinetic_energy_gradient_scheme, δx_u², δu²_stencil, u, v)
    
    ∂Kᴸ = (δKuᴸ + δKvˢ) / Δxᶠᶜᶜ(i, j, k, grid)
    ∂Kᴿ = (δKuᴿ + δKvˢ) / Δxᶠᶜᶜ(i, j, k, grid)

    return ifelse(û > 0, ∂Kᴸ, ∂Kᴿ)
end

@inline function bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantKineticEnergyUpwinding, u, v)

    @inbounds v̂ = v[i, j, k]

    δv²_stencil  = scheme.upwinding.δv²_stencil    
    cross_scheme = scheme.upwinding.cross_scheme
    δy_v²        = scheme.auxiliary_fields.δy_v²

    δKuˢ =    _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, cross_scheme, δy_u², u, v)
    δKvᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.kinetic_energy_gradient_scheme, δy_v², δv²_stencil, u, v) 
    δKvᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.kinetic_energy_gradient_scheme, δy_v², δv²_stencil, u, v) 
    
    ∂Kᴸ = (δKvᴸ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid) 
    ∂Kᴿ = (δKvᴿ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid)

    return ifelse(v̂ > 0, ∂Kᴸ, ∂Kᴿ)
end
