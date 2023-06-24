const VectorInvariantVelocityVerticalUpwinding  = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:VelocityUpwinding}

#####
##### Velocity upwinding is a Partial Upwinding where the upwind choice occurrs _inside_
##### the difference operator (i.e velocity upwinding) instead of outside (i.e., derivative upwinding).
##### _MOST_ stable formulation at the expense of a low kinetic energy
##### 

##### 
##### Velocity Upwinding of Divergence flux
#####

@inline function upwinded_Ax_uᶜᶜᶜ(i, j, k, grid, scheme, u) 
    û = ℑxᶜᵃᵃ(i, j, k, grid, u)

    Uᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ax_qᶠᶜᶜ, u)
    Uᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ax_qᶠᶜᶜ, u)

    return ifelse(û > 0, Uᴸ, Uᴿ)
end

@inline function upwinded_Ay_vᶜᶜᶜ(i, j, k, grid, scheme, v) 
    v̂ = ℑyᵃᶜᵃ(i, j, k, grid, v)

    Vᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ay_qᶜᶠᶜ, v)
    Vᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ay_qᶜᶠᶜ, v)

    return ifelse(v̂ > 0, Vᴸ, Vᴿ)
end

@inline reconstructed_Ax_uᶠᶠᶜ(i, j, k, grid, scheme, u) = 
     _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.upwinding.cross_scheme, Ax_qᶠᶜᶜ, u)

@inline reconstructed_Ay_vᶠᶠᶜ(i, j, k, grid, scheme, v) = 
     _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.upwinding.cross_scheme, Ay_qᶜᶠᶜ, v)

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantVelocityVerticalUpwinding, u, v) 
    @inbounds û = u[i, j, k] 
    
    δu = δxᶠᶜᶜ(i, j, k, grid,      upwinded_Ax_uᶜᶜᶜ, scheme, u) 
    δv = δyᶠᶜᶜ(i, j, k, grid, reconstructed_Ay_vᶠᶠᶜ, scheme, v)

    return û * (δu + δv)
end

@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantVelocityVerticalUpwinding, u, v) 
    @inbounds v̂ = v[i, j, k] 

    δu = δxᶜᵃᵃ(i, j, k, grid,      upwinded_Ax_uᶠᶠᶜ, scheme, u) 
    δv = δyᵃᶠᵃ(i, j, k, grid, reconstructed_Ay_vᶜᶜᶜ, scheme, v)

    return v̂ * (δu + δv)
end

##### 
##### Velocity Upwinding of Kinetic Energy gradient
#####

@inline function upwinded_u²ᶜᶜᶜ(i, j, k, grid, scheme, u) 
    û = ℑxᶜᵃᵃ(i, j, k, grid, u)

    Uᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², u)
    Uᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², u)

    return ifelse(û > 0, Uᴸ, Uᴿ)
end

@inline function upwinded_v²ᶜᶜᶜ(i, j, k, grid, scheme, v) 
    v̂ = ℑyᵃᶜᵃ(i, j, k, grid, v)

    Vᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², v)
    Vᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², v)

    return ifelse(v̂ > 0, Vᴸ, Vᴿ)
end

@inline reconstructed_u²ᶜᶜᶜ(i, j, k, grid, scheme, u) =
     _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.upwinding.cross_scheme, half_ϕ², u)

@inline reconstructed_v²ᶜᶜᶜ(i, j, k, grid, scheme, v) = 
     _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.upwinding.cross_scheme, half_ϕ², v)

@inline function bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantVelocityVerticalUpwinding, u, v)

    δKu = δxᶠᵃᵃ(i, j, k, grid,      upwinded_u²ᶜᶜᶜ, scheme, u)
    δKv = δxᶠᵃᵃ(i, j, k, grid, reconstructed_v²ᶜᶜᶜ, scheme, v)

    return (δKu + δKv) / Δxᶠᶜᶜ(i, j, k, grid)
end

@inline function bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantVelocityVerticalUpwinding, u, v)

    δKu = δyᵃᶠᵃ(i, j, k, grid,      upwinded_u²ᶜᶜᶜ, scheme, u)
    δKv = δyᵃᶠᵃ(i, j, k, grid, reconstructed_v²ᶜᶜᶜ, scheme, v)

    return (δKu + δKv) / Δyᶜᶠᶜ(i, j, k, grid)
end