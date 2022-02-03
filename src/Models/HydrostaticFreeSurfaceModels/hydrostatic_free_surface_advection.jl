using Oceananigans.Operators

using Oceananigans.Operators: Δx_qᶜᶠᶜ, Δy_qᶠᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ, Az_qᶜᶜᶜ
using Oceananigans.Advection: div_𝐯u, div_𝐯v

######
###### Horizontally-vector-invariant formulation of momentum advection
######
###### Follows https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#vector-invariant-momentum-equations
######

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2
@inline Khᶜᶜᶜ(i, j, k, grid, u, v) = (ℑxᶜᵃᵃ(i, j, k, grid, ϕ², u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², v)) / 2

@inbounds ζ₂wᶠᶜᶠ(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶜ, w) * ∂zᶠᶜᶠ(i, j, k, grid, u) / Azᶠᶜᶜ(i, j, k, grid)
@inbounds ζ₁wᶜᶠᶠ(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶜ, w) * ∂zᶜᶠᶠ(i, j, k, grid, v) / Azᶜᶠᶜ(i, j, k, grid)

@inline U_dot_∇u(i, j, k, grid, advection::VectorInvariant, U) = (
    - ℑyᵃᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, U.u, U.v) * ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_vᶜᶠᶜ, U.v) / Δxᶠᶜᶜ(i, j, k, grid) # Vertical relative vorticity term
    + ∂xᶠᶜᶜ(i, j, k, grid, Khᶜᶜᶜ, U.u, U.v)    # Bernoulli head term
    + ℑzᵃᵃᶜ(i, j, k, grid, ζ₂wᶠᶜᶠ, U.u, U.w))  # Horizontal vorticity / vertical advection term

@inline U_dot_∇v(i, j, k, grid, advection::VectorInvariant, U) = (
    + ℑxᶜᵃᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, U.u, U.v) * ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_uᶠᶜᶜ, U.u) / Δyᶜᶠᶜ(i, j, k, grid) # Vertical relative vorticity term
    + ∂yᶜᶠᶜ(i, j, k, grid, Khᶜᶜᶜ, U.u, U.v)   # Bernoulli head term
    + ℑzᵃᵃᶜ(i, j, k, grid, ζ₁wᶜᶠᶠ, U.v, U.w)) # Horizontal vorticity / vertical advection term

######
###### Conservative formulation of momentum advection
######

@inline U_dot_∇u(i, j, k, grid, advection::AbstractAdvectionScheme, U) = div_𝐯u(i, j, k, grid, advection, U, U.u)
@inline U_dot_∇v(i, j, k, grid, advection::AbstractAdvectionScheme, U) = div_𝐯v(i, j, k, grid, advection, U, U.v)

######
###### No advection
######

@inline U_dot_∇u(i, j, k, grid::AbstractGrid{FT}, advection::Nothing, U) where FT = zero(FT)
@inline U_dot_∇v(i, j, k, grid::AbstractGrid{FT}, advection::Nothing, U) where FT = zero(FT)
