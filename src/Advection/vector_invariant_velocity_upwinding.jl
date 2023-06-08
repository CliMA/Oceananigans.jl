const VectorInvariantVelocityVerticalUpwinding  = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:VelocityUpwinding}

#####
##### Velocity upwinding is a Partial Upwinding where the upwind choice occurrs _inside_
##### the difference operator (i.e velocity upwinding) instead of outside (i.e., derivative upwinding).
##### _MOST_ stable formulation at the expense of a low kinetic energy
##### 

##### 
##### Velocity Upwinding of Divergence flux
#####

@inline function Auᶜᶜᶜ(i, j, k, grid, scheme, u) 
    û  = ℑxᶜᵃᵃ(i, j, k, grid, u)
    Uᴿ = _upwind_interpolate_xᶜᵃᵃ(i, j, k, grid, û, scheme, scheme.vertical_scheme, Ax_qᶠᶜᶜ, u)
    return Uᴿ
end

@inline function Avᶜᶜᶜ(i, j, k, grid, scheme, v) 
    v̂  = ℑyᵃᶜᵃ(i, j, k, grid, v)
    Vᴿ = _upwind_interpolate_yᵃᶜᵃ(i, j, k, grid, v̂, scheme, scheme.vertical_scheme, Ay_qᶜᶠᶜ, v)
    return Vᴿ
end

@inline Auᶠᶠᶜ(i, j, k, grid, scheme, u) = 
     _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.upwinding_treatment.cross_scheme, Ax_qᶠᶜᶜ, u)

@inline Avᶠᶠᶜ(i, j, k, grid, scheme, v) = 
     _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.upwinding_treatment.cross_scheme, Ay_qᶜᶠᶜ, v)

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantVelocityVerticalUpwinding, u, v) 
    @inbounds û = u[i, j, k] 
    
    δu = δxᶠᶜᶜ(i, j, k, grid, Auᶜᶜᶜ, scheme, u) 
    δv = δyᶠᶜᶜ(i, j, k, grid, Avᶠᶠᶜ, scheme, v)

    return û * (δu + δv)
end

@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantVelocityVerticalUpwinding, u, v) 
    @inbounds v̂ = v[i, j, k] 

    δu = δxᶜᵃᵃ(i, j, k, grid, Auᶠᶠᶜ, scheme, u) 
    δv = δyᵃᶠᵃ(i, j, k, grid, Avᶜᶜᶜ, scheme, v)

    return v̂ * (δu + δv)
end

##### 
##### Velocity Upwinding of Kinetic Energy gradient
#####

@inline function uᵁ²ᶜᶜᶜ(i, j, k, grid, scheme, u) 
    û = ℑxᶜᵃᵃ(i, j, k, grid, u)
    Uᴿ = _upwind_interpolate_xᶜᵃᵃ(i, j, k, grid, û, scheme, scheme.vertical_scheme, half_ϕ², u)
    return Uᴿ
end

@inline function vᵁ²ᶜᶜᶜ(i, j, k, grid, scheme, v) 
    v̂  = ℑyᵃᶜᵃ(i, j, k, grid, v)
    Vᴿ = _upwind_interpolate_yᵃᶜᵃ(i, j, k, grid, v̂, scheme, scheme.vertical_scheme, half_ϕ², v)
    return Vᴿ
end

@inline uˢ²ᶜᶜᶜ(i, j, k, grid, scheme, u) =
     _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.upwinding_treatment.cross_scheme, half_ϕ², u)

@inline vˢ²ᶜᶜᶜ(i, j, k, grid, scheme, v) = 
     _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.upwinding_treatment.cross_scheme, half_ϕ², v)

@inline function bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantVelocityVerticalUpwinding, u, v)

    δKu = δxᶠᵃᵃ(i, j, k, grid, uᵁ²ᶜᶜᶜ, scheme, u)
    δKv = δxᶠᵃᵃ(i, j, k, grid, vˢ²ᶜᶜᶜ, scheme, v)

    return (δKu + δKv) / Δxᶠᶜᶜ(i, j, k, grid)
end

@inline function bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantVelocityVerticalUpwinding, u, v)

    δKu = δyᵃᶠᵃ(i, j, k, grid, uˢ²ᶜᶜᶜ, scheme, u)
    δKv = δyᵃᶠᵃ(i, j, k, grid, vᵁ²ᶜᶜᶜ, scheme, v)

    return (δKu + δKv) / Δyᶜᶠᶜ(i, j, k, grid)
end