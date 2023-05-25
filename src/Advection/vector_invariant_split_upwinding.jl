const VectorInvariantSplitVerticalUpwinding  = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:SplitUpwinding}

#####
##### Split upwinding is a Partial Upwinding where the upwind choice occurrs _inside_
##### the difference operator instead of outside. This allows angular momentum conservation.
##### 

##### 
##### Split Upwinding of Divergence flux (untested yet)
#####

@inline function Auᶜᶜᶜ(i, j, k, grid, scheme, u) 
    û = ℑxᶜᵃᵃ(i, j, k, grid, u)

    Uᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ax_qᶠᶜᶜ, u)
    Uᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ax_qᶠᶜᶜ, u)

    return ifelse(û > 0, Uᴸ, Uᴿ)
end

@inline function Avᶜᶜᶜ(i, j, k, grid, scheme, v) 
    v̂ = ℑyᵃᶜᵃ(i, j, k, grid, v)

    Vᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ay_qᶜᶠᶜ, v)
    Vᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ay_qᶜᶠᶜ, v)

    return ifelse(v̂ > 0, Vᴸ, Vᴿ)
end

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantSplitVerticalUpwinding, u, v) 
    @inbounds û = u[i, j, k] 
    
    δu = δxᶠᵃᵃ(i, j, k, grid, scheme, u) 
    δv = _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δ_V, u, v)

    return û * (δu + δv)
end


@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantSplitVerticalUpwinding, u, v) 
    @inbounds v̂ = v[i, j, k] 

    δu = _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δ_U, u, v) 
    δv = δyᵃᶠᵃ(i, j, k, grid, scheme, v)

    return v̂ * (δu + δv)
end

##### 
##### Split Upwinding of Bernoulli term
#####

@inline function u²ᶜᶜᶜ(i, j, k, grid, scheme, u) 
    û = ℑxᶜᵃᵃ(i, j, k, grid, u)

    Uᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², u)
    Uᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², u)

    return ifelse(û > 0, Uᴸ, Uᴿ)
end

@inline function v²ᶜᶜᶜ(i, j, k, grid, scheme, v) 
    v̂ = ℑyᵃᶜᵃ(i, j, k, grid, v)

    Vᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², v)
    Vᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², v)

    return ifelse(v̂ > 0, Vᴸ, Vᴿ)
end

@inline function bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantSplitVerticalUpwinding, u, v)

    δKv = _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δxᶠᵃᵃ, half_ϕ², v) 
    δKu =  δxᶠᵃᵃ(i, j, k, grid, u²ᶜᶜᶜ, scheme, u)

    return (δKu + δKv) / Δxᶠᶜᶜ(i, j, k, grid)
end

@inline function bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantSplitVerticalUpwinding, u, v)

    δKu = _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, δyᵃᶠᵃ, half_ϕ², u)
    δKv =  δyᵃᶠᵃ(i, j, k, grid, v²ᶜᶜᶜ, scheme, v)

    return (δKu + δKv) / Δyᶜᶠᶜ(i, j, k, grid)
end