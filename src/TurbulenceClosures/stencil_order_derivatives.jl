"""
    stencil_order(closure)

Return `Val(order)`, where `order` is the order of accuracy of the stencils used to compute the first derivatives in the
diffusive and viscous fluxes of `closure`. `order` is the order of the discrete Laplacian (this is not biharmonic diffusion).
"""
@inline stencil_order(closure) = Val(2)

##### Viscous "stresses" τᵢⱼ such that the viscous flux of uᵢ in direction j is -ν τᵢⱼ.
#####
##### τᵢⱼ = ∂ⱼuᵢ + ∂ᵢuⱼ, with ∂ⱼuᵢ at the closure's stencil order and the "transpose" term ∂ᵢuⱼ at second order.
##### For constant ν, the transpose term contributes ν ∂ᵢ(∂ⱼuⱼ) to the momentum tendency, which vanishes
##### identically for a velocity field that is divergence-free in the (second-order) discrete sense
##### enforced by the pressure solver. The viscous term is then the higher-order Laplacian ν ∇²uᵢ.
#####

@inline stress_ux(i, j, k, grid, order, u, v, w) = ∂xᵣᶜᶜᶜ(i, j, k, grid, order, u) + ∂x_u(i, j, k, grid, u)
@inline stress_vx(i, j, k, grid, order, u, v, w) = ∂xᵣᶠᶠᶜ(i, j, k, grid, order, v) + ∂y_u(i, j, k, grid, u)
@inline stress_wx(i, j, k, grid, order, u, v, w) = ∂xᵣᶠᶜᶠ(i, j, k, grid, order, w) + ∂z_u(i, j, k, grid, u)
@inline stress_uy(i, j, k, grid, order, u, v, w) = ∂yᵣᶠᶠᶜ(i, j, k, grid, order, u) + ∂x_v(i, j, k, grid, v)
@inline stress_vy(i, j, k, grid, order, u, v, w) = ∂yᵣᶜᶜᶜ(i, j, k, grid, order, v) + ∂y_v(i, j, k, grid, v)
@inline stress_wy(i, j, k, grid, order, u, v, w) = ∂yᵣᶜᶠᶠ(i, j, k, grid, order, w) + ∂z_v(i, j, k, grid, v)
@inline stress_uz(i, j, k, grid, order, u, v, w) = ∂zᶠᶜᶠ(i, j, k, grid, order, u) + ∂x_w(i, j, k, grid, w)
@inline stress_vz(i, j, k, grid, order, u, v, w) = ∂zᶜᶠᶠ(i, j, k, grid, order, v) + ∂y_w(i, j, k, grid, w)
@inline stress_wz(i, j, k, grid, order, u, v, w) = ∂zᶜᶜᶜ(i, j, k, grid, order, w) + ∂z_w(i, j, k, grid, w)
