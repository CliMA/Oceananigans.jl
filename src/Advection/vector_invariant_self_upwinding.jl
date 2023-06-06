const VectorInvariantSelfVerticalUpwinding = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:OnlySelfUpwinding}

##### 
##### Self Upwinding of Divergence Flux, the best option!
#####

@inline δx_U(i, j, k, grid, u, v) =  δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline δy_V(i, j, k, grid, u, v) =  δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# Velocity smoothness for divergence upwinding
@inline U_smoothness(i, j, k, grid, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, u)
@inline V_smoothness(i, j, k, grid, u, v) = ℑyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, v)

# Divergence smoothness for divergence upwinding
@inline divergence_smoothness(i, j, k, grid, u, v) = δx_U(i, j, k, grid, u, v) + δy_V(i, j, k, grid, u, v)

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, u, v)

    @inbounds û = u[i, j, k]
    δU_stencil   = scheme.upwinding_treatment.δU_stencil    
    cross_scheme = scheme.upwinding_treatment.cross_scheme
    side         = upwinding_direction(û)

    δu =    _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.vertical_scheme, side, δx_U, δU_stencil, u, v) 
    δv = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, cross_scheme, δy_V, u, v) 

    return û * (δv + δu)
end

@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantSelfVerticalUpwinding, u, v)
    
    @inbounds v̂ = v[i, j, k]
    δV_stencil   = scheme.upwinding_treatment.δV_stencil
    cross_scheme = scheme.upwinding_treatment.cross_scheme
    side         = upwinding_direction(v̂)

    δv =    _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.vertical_scheme, side, δy_V, δV_stencil, u, v) 
    δu = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, cross_scheme, δx_U, u, v)

    return v̂ * (δv + δu)
end

#####
##### Self Upwinding of Kinetic Energy Gradient 
#####

const VectorInvariantVerticalUpwinding = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme}

@inline half_ϕ²(i, j, k, grid, ϕ) = ϕ[i, j, k]^2 / 2

@inline δx_u²(i, j, k, grid, u, v) = δxᶜᵃᵃ(i, j, k, grid, half_ϕ², u)
@inline δy_u²(i, j, k, grid, u, v) = δyᶠᶠᶜ(i, j, k, grid, half_ϕ², u)

@inline δx_v²(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, half_ϕ², v)
@inline δy_v²(i, j, k, grid, u, v) = δyᵃᶜᵃ(i, j, k, grid, half_ϕ², v)

@inline u_smoothness(i, j, k, grid, u, v) = ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline v_smoothness(i, j, k, grid, u, v) = ℑyᵃᶜᵃ(i, j, k, grid, v)

@inline function bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantVerticalUpwinding, u, v)

    @inbounds û = u[i, j, k]
    δu²_stencil  = scheme.upwinding_treatment.δu²_stencil    
    cross_scheme = scheme.upwinding_treatment.cross_scheme
    side         = upwinding_direction(û)

    δKu =    _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.vertical_scheme, side, δx_u², δu²_stencil, u, v)
    δKv = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, cross_scheme, δx_v², u, v)
    
    return (δKu + δKv) / Δxᶠᶜᶜ(i, j, k, grid)
end

@inline function bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantVerticalUpwinding, u, v)

    @inbounds v̂ = v[i, j, k]
    δv²_stencil  = scheme.upwinding_treatment.δv²_stencil    
    cross_scheme = scheme.upwinding_treatment.cross_scheme
    side         = upwinding_direction(v̂)

    δKv =    _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.vertical_scheme, side, δy_v², δv²_stencil, u, v) 
    δKu = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, cross_scheme, δy_u², u, v)
    
    return (δKu + δKv) / Δyᶜᶠᶜ(i, j, k, grid) 
end