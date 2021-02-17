using Oceananigans.Advection: div_Uu, div_Uv
using Oceananigans.Operators

######
###### Horizontally-vector-invariant formulation of momentum advection
######

@inline Δy_vᶜᶠᶜ(i, j, k, grid, v) = @inbounds Δy(i, j, k, grid) * v[i, j, k]
@inline Δx_uᶠᶜᶜ(i, j, k, grid, u) = @inbounds Δx(i, j, k, grid) * u[i, j, k]

@inline Axyᶠᶠᵃ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

@inline Γᶠᶠᶜ(i, j, k, grid, u, v) = δxᶠᵃᵃ(i, j, k, grid, Δy_vᶜᶠᶜ, v) - δyᵃᶠᵃ(i, j, k, grid, Δx_uᶠᶜᶜ, u)
@inline ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) = Γᶠᶠᶜ(i, j, k, grid, u, v) / Axyᶠᶠᵃ(i, j, k, grid)

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2
@inline Khᶜᶜᶜ(i, j, k, grid, u, v) = (ℑxᶜᵃᵃ(i, j, k, grid, ϕ², u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², v)) / 2

@inbounds ζ₂wᶠᶜᶠ(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, w) * δzᵃᵃᶠ(i, j, k, grid, u) / ΔzC(i, j, k, grid)
@inbounds ζ₁wᶜᶠᶠ(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, w) * δzᵃᵃᶠ(i, j, k, grid, v) / ΔzC(i, j, k, grid)

@inline U_dot_∇u(i, j, k, grid, advection::VectorInvariant, U) = (
    - ℑyᵃᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, U.u, U.v) * ℑxyᶠᶜᵃ(i, j, k, grid, U.v)
    + δxᶠᵃᵃ(i, j, k, grid, Khᶜᶜᶜ, U.u, U.v) / Δx(i, j, k, grid)
    + ℑzᵃᵃᶜ(i, j, k, grid, ζ₂wᶠᶜᶠ, U.u, U.w))

@inline U_dot_∇v(i, j, k, grid, advection::VectorInvariant, U) = (
    + ℑxᶜᵃᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, U.u, U.v) * ℑxyᶜᶠᵃ(i, j, k, grid, U.u)
    + δyᵃᶠᵃ(i, j, k, grid, Khᶜᶜᶜ, U.u, U.v) / Δy(i, j, k, grid)
    + ℑzᵃᵃᶜ(i, j, k, grid, ζ₁wᶜᶠᶠ, U.v, U.w))

######
###### Conservative formulation of momentum advection
######

@inline U_dot_∇u(i, j, k, grid, advection::AbstractAdvectionScheme, U) = div_Uu(i, j, k, grid, advection, U, U.u)
@inline U_dot_∇v(i, j, k, grid, advection::AbstractAdvectionScheme, U) = div_Uv(i, j, k, grid, advection, U, U.v)

