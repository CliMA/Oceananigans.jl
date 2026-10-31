#####
##### Derivatives used by diffusive and viscous fluxes, selected by the closure's stencil order
#####
##### `ScalarDiffusivity(order=4)` computes fluxes with the fourth-order-accurate first derivatives
##### `fourth_order_∂xᵣᶠᶜᶜ` etc. (see Operators/fourth_order_derivative_operators.jl), whose flux divergence
##### is the fourth-order five-point Laplacian and which fall back to second order next to walls and immersed
##### boundaries. With `order = 2` the fluxes use the second-order derivatives `∂xᵣᶠᶜᶜ` etc., as always.
#####

using Oceananigans.Operators: ∂xᵣᶠᶜᶜ, ∂xᵣᶜᶜᶜ, ∂xᵣᶠᶠᶜ, ∂xᵣᶠᶜᶠ,
                              ∂yᵣᶜᶠᶜ, ∂yᵣᶠᶠᶜ, ∂yᵣᶜᶜᶜ, ∂yᵣᶜᶠᶠ,
                              ∂zᶜᶜᶠ, ∂zᶠᶜᶠ, ∂zᶜᶠᶠ, ∂zᶜᶜᶜ

"""
    stencil_order(closure)

Return `Val(order)`, where `order` is the order of accuracy of the stencils (2 or 4) used to compute
the first derivatives in the diffusive and viscous fluxes of `closure`. With `Val(4)` the discrete
Laplacian is the fourth-order five-point Laplacian (this is not biharmonic diffusion).
"""
@inline stencil_order(closure) = Val(2)

for (∂, ∂², ∂⁽⁴⁾) in ((:diffusive_∂xᶠᶜᶜ,   :∂xᵣᶠᶜᶜ, :fourth_order_∂xᵣᶠᶜᶜ), # tracers
                     (:diffusive_∂yᶜᶠᶜ,   :∂yᵣᶜᶠᶜ, :fourth_order_∂yᵣᶜᶠᶜ),
                     (:diffusive_∂zᶜᶜᶠ,   :∂zᶜᶜᶠ,  :fourth_order_∂zᶜᶜᶠ),
                     (:diffusive_∂xᶜᶜᶜ,   :∂xᵣᶜᶜᶜ, :fourth_order_∂xᵣᶜᶜᶜ), # u
                     (:diffusive_∂y_uᶠᶠᶜ, :∂yᵣᶠᶠᶜ, :fourth_order_∂yᵣᶠᶠᶜ),
                     (:diffusive_∂z_uᶠᶜᶠ, :∂zᶠᶜᶠ,  :fourth_order_∂zᶠᶜᶠ),
                     (:diffusive_∂x_vᶠᶠᶜ, :∂xᵣᶠᶠᶜ, :fourth_order_∂xᵣᶠᶠᶜ), # v
                     (:diffusive_∂yᶜᶜᶜ,   :∂yᵣᶜᶜᶜ, :fourth_order_∂yᵣᶜᶜᶜ),
                     (:diffusive_∂z_vᶜᶠᶠ, :∂zᶜᶠᶠ,  :fourth_order_∂zᶜᶠᶠ),
                     (:diffusive_∂x_wᶠᶜᶠ, :∂xᵣᶠᶜᶠ, :fourth_order_∂xᵣᶠᶜᶠ), # w
                     (:diffusive_∂y_wᶜᶠᶠ, :∂yᵣᶜᶠᶠ, :fourth_order_∂yᵣᶜᶠᶠ),
                     (:diffusive_∂zᶜᶜᶜ,   :∂zᶜᶜᶜ,  :fourth_order_∂zᶜᶜᶜ))
    @eval begin
        @inline $∂(i, j, k, grid, ::Val{2}, φ) = $∂²(i, j, k, grid, φ)
        @inline $∂(i, j, k, grid, ::Val{4}, φ) = Operators.$∂⁽⁴⁾(i, j, k, grid, φ)
    end
end

##### Viscous "stresses" τᵢⱼ such that the viscous flux of uᵢ in direction j is -ν τᵢⱼ.
#####
##### With second-order stencils, τᵢⱼ = 2Σᵢⱼ = ∂ⱼuᵢ + ∂ᵢuⱼ. With fourth-order stencils we compute
##### ∂ⱼuᵢ with the fourth-order derivative and keep the "transpose" term ∂ᵢuⱼ second order.
##### For constant ν, the transpose term contributes ν ∂ᵢ(∂ⱼuⱼ) to the momentum tendency, which vanishes
##### identically for a velocity field that is divergence-free in the (second-order) discrete sense
##### enforced by the pressure solver. The viscous term is then the fourth-order Laplacian ν ∇²uᵢ.
##### Differentiating the transpose term at fourth order would instead leave a spurious term of
##### the size of the truncation error of the second-order Laplacian.
#####

@inline stress_ux(i, j, k, grid, ::Val{2}, u, v, w) = 2 * Σ₁₁(i, j, k, grid, u, v, w)
@inline stress_vx(i, j, k, grid, ::Val{2}, u, v, w) = 2 * Σ₂₁(i, j, k, grid, u, v, w)
@inline stress_wx(i, j, k, grid, ::Val{2}, u, v, w) = 2 * Σ₃₁(i, j, k, grid, u, v, w)
@inline stress_uy(i, j, k, grid, ::Val{2}, u, v, w) = 2 * Σ₁₂(i, j, k, grid, u, v, w)
@inline stress_vy(i, j, k, grid, ::Val{2}, u, v, w) = 2 * Σ₂₂(i, j, k, grid, u, v, w)
@inline stress_wy(i, j, k, grid, ::Val{2}, u, v, w) = 2 * Σ₃₂(i, j, k, grid, u, v, w)
@inline stress_uz(i, j, k, grid, ::Val{2}, u, v, w) = 2 * Σ₁₃(i, j, k, grid, u, v, w)
@inline stress_vz(i, j, k, grid, ::Val{2}, u, v, w) = 2 * Σ₂₃(i, j, k, grid, u, v, w)
@inline stress_wz(i, j, k, grid, ::Val{2}, u, v, w) = 2 * Σ₃₃(i, j, k, grid, u, v, w)

@inline stress_ux(i, j, k, grid, order::Val{4}, u, v, w) = diffusive_∂xᶜᶜᶜ(i, j, k, grid, order, u)   + ∂x_u(i, j, k, grid, u)
@inline stress_vx(i, j, k, grid, order::Val{4}, u, v, w) = diffusive_∂x_vᶠᶠᶜ(i, j, k, grid, order, v) + ∂y_u(i, j, k, grid, u)
@inline stress_wx(i, j, k, grid, order::Val{4}, u, v, w) = diffusive_∂x_wᶠᶜᶠ(i, j, k, grid, order, w) + ∂z_u(i, j, k, grid, u)
@inline stress_uy(i, j, k, grid, order::Val{4}, u, v, w) = diffusive_∂y_uᶠᶠᶜ(i, j, k, grid, order, u) + ∂x_v(i, j, k, grid, v)
@inline stress_vy(i, j, k, grid, order::Val{4}, u, v, w) = diffusive_∂yᶜᶜᶜ(i, j, k, grid, order, v)   + ∂y_v(i, j, k, grid, v)
@inline stress_wy(i, j, k, grid, order::Val{4}, u, v, w) = diffusive_∂y_wᶜᶠᶠ(i, j, k, grid, order, w) + ∂z_v(i, j, k, grid, v)
@inline stress_uz(i, j, k, grid, order::Val{4}, u, v, w) = diffusive_∂z_uᶠᶜᶠ(i, j, k, grid, order, u) + ∂x_w(i, j, k, grid, w)
@inline stress_vz(i, j, k, grid, order::Val{4}, u, v, w) = diffusive_∂z_vᶜᶠᶠ(i, j, k, grid, order, v) + ∂y_w(i, j, k, grid, w)
@inline stress_wz(i, j, k, grid, order::Val{4}, u, v, w) = diffusive_∂zᶜᶜᶜ(i, j, k, grid, order, w)   + ∂z_w(i, j, k, grid, w)
