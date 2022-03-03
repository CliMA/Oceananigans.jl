using Oceananigans.Operators
using Oceananigans.Fields: ZeroField

using Oceananigans.Operators: Δx_qᶜᶠᶜ, Δy_qᶠᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ, Az_qᶜᶜᶜ
using Oceananigans.Advection:
      _advective_momentum_flux_Uu,
      _advective_momentum_flux_Vv,
      upwind_biased_product

import Oceananigans.Advection:
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

@inbounds ζ₂wᶠᶜᶠ(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶜ, w) * ∂zᶠᶜᶠ(i, j, k, grid, u) / Azᶠᶜᶜ(i, j, k, grid)
@inbounds ζ₁wᶜᶠᶠ(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶜ, w) * ∂zᶜᶠᶠ(i, j, k, grid, v) / Azᶜᶠᶜ(i, j, k, grid)

@inline U_dot_∇u(i, j, k, grid, scheme::VectorInvariantSchemes, U) = (
    + vertical_vorticity_U(i, j, k, grid, scheme, U.u, U.v)  # Vertical relative vorticity term
    + bernoulli_head_U(i, j, k, grid, scheme, U.u, U.v) # Bernoulli head term
    + ℑzᵃᵃᶜ(i, j, k, grid, ζ₂wᶠᶜᶠ, U.u, U.w))  # Horizontal vorticity / vertical advection term

@inline U_dot_∇v(i, j, k, grid, scheme::VectorInvariantSchemes, U) = (
    + vertical_vorticity_V(i, j, k, grid, scheme, U.u, U.v) # Vertical relative vorticity term
    + bernoulli_head_V(i, j, k, grid, scheme, U.u, U.v) # Bernoulli head term
    + ℑzᵃᵃᶜ(i, j, k, grid, ζ₁wᶜᶠᶠ, U.v, U.w)) # Horizontal vorticity / vertical advection term

@inline vertical_vorticity_U(i, j, k, grid, ::VectorInvariant, u, v) = - ℑyᵃᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
@inline vertical_vorticity_V(i, j, k, grid, ::VectorInvariant, u, v) = + ℑxᶜᵃᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)

@inline bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantSchemes, u, v) = ∂xᶠᶜᶜ(i, j, k, grid, Khᶜᶜᶜ, scheme, u, v)    
@inline bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantSchemes, u, v) = ∂yᶜᶠᶜ(i, j, k, grid, Khᶜᶜᶜ, scheme, u, v)  

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2
@inline Khᶜᶜᶜ(i, j, k, grid, ::VectorInvariantSchemes, u, v) = (ℑxᶜᵃᵃ(i, j, k, grid, ϕ², u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², v)) / 2

@inline function vertical_vorticity_U(i, j, k, grid, scheme::WENOVectorInvariant, u, v)
    v̂  =  ℑxyᶠᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)
    ζᴸ =  left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, ζ₃ᶠᶠᶜ, u, v)
    ζᴿ = right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, ζ₃ᶠᶠᶜ, u, v)
    return - upwind_biased_product(v̂, ζᴸ, ζᴿ) / Δxᶠᶜᶜ(i, j, k, grid) 
end

@inline function vertical_vorticity_V(i, j, k, grid, scheme::WENOVectorInvariant, u, v)
    û  =  ℑxyᶜᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)
    ζᴸ =  left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, ζ₃ᶠᶠᶜ, u, v)
    ζᴿ = right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, ζ₃ᶠᶠᶜ, u, v)
    return + upwind_biased_product(û, ζᴸ, ζᴿ) / Δyᶜᶠᶜ(i, j, k, grid)
end

######
###### Conservative formulation of momentum advection
######

@inline U_dot_∇u(i, j, k, grid, scheme::AbstractAdvectionScheme, U) = div_𝐯u(i, j, k, grid, scheme, U, U.u)
@inline U_dot_∇v(i, j, k, grid, scheme::AbstractAdvectionScheme, U) = div_𝐯v(i, j, k, grid, scheme, U, U.v)

@inline div_𝐯u(i, j, k, grid, scheme::VectorInvariantSchemes, U, u) = U_dot_∇u(i, j, k, grid, scheme, U)
@inline div_𝐯v(i, j, k, grid, scheme::VectorInvariantSchemes, U, v) = U_dot_∇v(i, j, k, grid, scheme, U)

@inline div_𝐯u(i, j, k, grid, scheme::VectorInvariantSchemes, U, ::ZeroField) = zero(eltype(grid))
@inline div_𝐯v(i, j, k, grid, scheme::VectorInvariantSchemes, U, ::ZeroField) = zero(eltype(grid))
@inline div_𝐯w(i, j, k, grid, scheme::VectorInvariantSchemes, U, ::ZeroField) = zero(eltype(grid))

@inline div_𝐯w(i, j, k, grid, scheme::VectorInvariantSchemes, U, w) = zero(eltype(grid))


######
###### No advection
######

@inline U_dot_∇u(i, j, k, grid::AbstractGrid{FT}, scheme::Nothing, U) where FT = zero(FT)
@inline U_dot_∇v(i, j, k, grid::AbstractGrid{FT}, scheme::Nothing, U) where FT = zero(FT)
