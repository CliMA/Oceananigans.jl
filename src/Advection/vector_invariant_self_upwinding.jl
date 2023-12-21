##### 
##### Self Upwinding of Divergence Flux, the best option!
#####

@inline δx_U(i, j, k, grid, u, v) =  δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline δy_V(i, j, k, grid, u, v) =  δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

@inline δx_U_plus_metric(i, j, k, grid, u, v) =  δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline δy_V_plus_metric(i, j, k, grid, u, v) =  δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# Velocity smoothness for divergence upwinding
@inline U_smoothness(i, j, k, grid, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline V_smoothness(i, j, k, grid, u, v) = ℑyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# Divergence smoothness for divergence upwinding
@inline divergence_smoothness(i, j, k, grid, u, v) = δx_U(i, j, k, grid, u, v) + δy_V(i, j, k, grid, u, v)

# Metric term for moving grids
metric_term(i, j, k, grid) = zero(grid)

@inline function upwinded_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, u, v)

    δU_stencil   = scheme.upwinding.δU_stencil    
    cross_scheme = scheme.upwinding.cross_scheme

    @inbounds û = u[i, j, k]
    δvˢ =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, cross_scheme, δy_V, u, v) 
    δuᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, δx_U, δU_stencil, u, v) 
    δuᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, δx_U, δU_stencil, u, v) 

    return upwind_biased_product(û, δuᴸ, δuᴿ) + û * (δvˢ + ℑxᶠᵃᵃ(i, j, k, grid, metric_term))
end

@inline function upwinded_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, u, v)
    
    δV_stencil   = scheme.upwinding.δV_stencil
    cross_scheme = scheme.upwinding.cross_scheme

    @inbounds v̂ = v[i, j, k]
    δuˢ =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, cross_scheme, δx_U, u, v)
    δvᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, δy_V, δV_stencil, u, v) 
    δvᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.divergence_scheme, δy_V, δV_stencil, u, v) 

    return upwind_biased_product(v̂, δvᴸ, δvᴿ) + v̂ * (δuˢ + ℑyᵃᶠᵃ(i, j, k, grid, metric_term))
end

#####
##### Self Upwinding of Kinetic Energy Gradient 
#####

@inline half_ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2 / 2

@inline δx_u²(i, j, k, grid, u, v) = δxᶜᵃᵃ(i, j, k, grid, half_ϕ², u)
@inline δy_u²(i, j, k, grid, u, v) = δyᶠᶠᶜ(i, j, k, grid, half_ϕ², u)

@inline δx_v²(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, half_ϕ², v)
@inline δy_v²(i, j, k, grid, u, v) = δyᵃᶜᵃ(i, j, k, grid, half_ϕ², v)

@inline u_smoothness(i, j, k, grid, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline v_smoothness(i, j, k, grid, u, v) = ℑyᵃᶜᵃ(i, j, k, grid, v)

@inline function bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantKineticEnergyUpwinding, u, v)

    @inbounds û = u[i, j, k]

    δu²_stencil  = scheme.upwinding.δu²_stencil    
    cross_scheme = scheme.upwinding.cross_scheme

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

    δKuˢ =    _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, cross_scheme, δy_u², u, v)
    δKvᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.kinetic_energy_gradient_scheme, δy_v², δv²_stencil, u, v) 
    δKvᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.kinetic_energy_gradient_scheme, δy_v², δv²_stencil, u, v) 
    
    ∂Kᴸ = (δKvᴸ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid) 
    ∂Kᴿ = (δKvᴿ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid)

    return ifelse(v̂ > 0, ∂Kᴸ, ∂Kᴿ)
end
