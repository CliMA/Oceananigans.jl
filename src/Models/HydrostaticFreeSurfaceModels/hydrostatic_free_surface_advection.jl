using Oceananigans.Operators
using Oceananigans.Operators: hack_sind

using Oceananigans.Advection:
      _advective_momentum_flux_Uu,
      _advective_momentum_flux_Vv,
      upwind_biased_product,
      div_𝐯u,
      div_𝐯v,
      div_𝐯w,
      left_biased_interpolate_xᶜᵃᵃ,
      right_biased_interpolate_xᶜᵃᵃ,
      left_biased_interpolate_yᵃᶜᵃ,
      right_biased_interpolate_yᵃᶜᵃ

######
###### Horizontally-vector-invariant formulation of momentum scheme
######
###### Follows https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#vector-invariant-momentum-equations
######

@inbounds ζ₂wᶠᶜᶠ(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶠᶜᶠ(i, j, k, grid, u) / Azᶠᶜᶜ(i, j, k, grid)
@inbounds ζ₁wᶜᶠᶠ(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶜᶠᶠ(i, j, k, grid, v) / Azᶜᶠᶜ(i, j, k, grid)

@inline U_dot_∇u(i, j, k, grid, scheme::VectorInvariantSchemes, U) = (
    + vertical_vorticity_U(i, j, k, grid, scheme, U.u, U.v)  # Vertical relative vorticity term
    + vertical_advection_U(i, j, k, grid, scheme, U.u, U.w)  # Horizontal vorticity / vertical advection term
    + bernoulli_head_U(i, j, k, grid, scheme, U.u, U.v)) # Bernoulli head term
    
@inline U_dot_∇v(i, j, k, grid, scheme::VectorInvariantSchemes, U) = (
    + vertical_vorticity_V(i, j, k, grid, scheme, U.u, U.v) # Vertical relative vorticity term
    + vertical_advection_V(i, j, k, grid, scheme, U.v, U.w)  # Horizontal vorticity / vertical advection term
    + bernoulli_head_V(i, j, k, grid, scheme, U.u, U.v)) # Bernoulli head term

@inline vertical_advection_U(i, j, k, grid, ::VectorInvariant, u, w) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₂wᶠᶜᶠ, u, w)
@inline vertical_advection_V(i, j, k, grid, ::VectorInvariant, v, w) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₁wᶜᶠᶠ, v, w)

@inline function vertical_advection_term_U(i, j, k, grid, scheme::WENOVectorInvariant, u, w)
    ŵ = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, Az_qᶜᶜᶠ, w) / Azᶠᶜᶜ(i, j, k, grid)
    ζᴸ =  left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, ∂zᶠᶜᶠ, u)
    ζᴿ = right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, ∂zᶠᶜᶠ, u)
    return upwind_biased_product(ŵ, ζᴸ, ζᴿ) 
end

@inline function vertical_advection_term_V(i, j, k, grid, scheme::WENOVectorInvariant, v, w)
    ŵ = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, Az_qᶜᶜᶠ, w) / Azᶜᶠᶜ(i, j, k, grid)
    ζᴸ =  left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, ∂zᶜᶠᶠ, v)
    ζᴿ = right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, ∂zᶜᶠᶠ, v)
    return upwind_biased_product(ŵ, ζᴸ, ζᴿ) 
end

@inline vertical_vorticity_U(i, j, k, grid, ::VectorInvariant, u, v) = - ℑyᵃᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
@inline vertical_vorticity_V(i, j, k, grid, ::VectorInvariant, u, v) = + ℑxᶜᵃᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)

@inline bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantSchemes, u, v) = ∂xᶠᶜᶜ(i, j, k, grid, Khᶜᶜᶜ, scheme, u, v)    
@inline bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantSchemes, u, v) = ∂yᶜᶠᶜ(i, j, k, grid, Khᶜᶜᶜ, scheme, u, v)  

@inline ζₜ(i, j, k, grid, u, v) = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) 

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2
@inline Khᶜᶜᶜ(i, j, k, grid, ::VectorInvariantSchemes, u, v) = (ℑxᶜᵃᵃ(i, j, k, grid, ϕ², u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², v)) / 2

@inline function vertical_vorticity_U(i, j, k, grid, scheme::WENOVectorInvariant, u, v)
    v̂  =  ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
    ζᴸ =  left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, ζₜ, u, v)
    ζᴿ = right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, ζₜ, u, v)
    return - upwind_biased_product(v̂, ζᴸ, ζᴿ) 
end

@inline function vertical_vorticity_V(i, j, k, grid, scheme::WENOVectorInvariant, u, v)
    û  =  ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)
    ζᴸ =  left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, ζₜ, u, v)
    ζᴿ = right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, ζₜ, u, v)
    return + upwind_biased_product(û, ζᴸ, ζᴿ) 
end

######
###### Conservative formulation of momentum advection
######

@inline U_dot_∇u(i, j, k, grid, scheme::AbstractAdvectionScheme, U) = div_𝐯u(i, j, k, grid, scheme, U, U.u)
@inline U_dot_∇v(i, j, k, grid, scheme::AbstractAdvectionScheme, U) = div_𝐯v(i, j, k, grid, scheme, U, U.v)

######
###### No advection
######

@inline U_dot_∇u(i, j, k, grid::AbstractGrid{FT}, scheme::Nothing, U) where FT = zero(FT)
@inline U_dot_∇v(i, j, k, grid::AbstractGrid{FT}, scheme::Nothing, U) where FT = zero(FT)
