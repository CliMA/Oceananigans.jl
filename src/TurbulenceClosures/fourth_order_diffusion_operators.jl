#####
##### Derivatives for second- and fourth-order diffusive fluxes
#####
##### `ScalarDiffusivity(order=4)` computes diffusive and viscous fluxes with the face derivative
#####
#####     ∂ᶠφ ≈ (15 (φᵢ - φᵢ₋₁) - (φᵢ₊₁ - φᵢ₋₂)) / 12Δ ,
#####
##### whose flux divergence, (Fᵢ₊₁ - Fᵢ) / Δ, is the fourth-order five-point Laplacian
#####
#####     ∂²φ ≈ (- φᵢ₋₂ + 16 φᵢ₋₁ - 30 φᵢ + 16 φᵢ₊₁ - φᵢ₊₂) / 12Δ² .
#####
##### Note that the fourth-order-accurate staggered derivative (27 (φᵢ - φᵢ₋₁) - (φᵢ₊₁ - φᵢ₋₂)) / 24Δ
##### yields a Laplacian that is only second-order accurate when differenced by the (second-order)
##### flux divergence: its error is half that of the three-point Laplacian, but has the same order.
#####
##### The fourth-order stencil is used only in directions in which the grid is a regularly-spaced
##### `RectilinearGrid`, and only when all four nodes of the stencil are active. Next to `Bounded` walls and
##### immersed boundaries the fluxes fall back on the second-order derivative, similar to the
##### reduction of advection stencils near boundaries. Fluxes through boundary faces are therefore
##### unchanged and boundary conditions are imposed exactly as for second-order diffusion.
#####

using Oceananigans.Grids: XRegularRG, YRegularRG, ZRegularRG, Flat, inactive_node, topology
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, PartialCellVerticalBoundaries
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ,
                              ∂xᵣᶠᶜᶜ, ∂xᵣᶜᶜᶜ, ∂xᵣᶠᶠᶜ, ∂xᵣᶠᶜᶠ,
                              ∂yᵣᶜᶠᶜ, ∂yᵣᶠᶠᶜ, ∂yᵣᶜᶜᶜ, ∂yᵣᶜᶠᶠ,
                              ∂zᶜᶜᶠ, ∂zᶠᶜᶠ, ∂zᶜᶠᶠ, ∂zᶜᶜᶜ

"""
    diffusion_order(closure)

Return `Val(order)`, where `order` is the order of accuracy of the discrete Laplacian
used to compute the diffusive and viscous fluxes of `closure`.
"""
@inline diffusion_order(closure) = Val(2)

# The order that is actually used in each direction: fourth order requires regular spacing.
@inline x_diffusion_order(grid, order) = Val(2)
@inline y_diffusion_order(grid, order) = Val(2)
@inline z_diffusion_order(grid, order) = Val(2)

@inline x_diffusion_order(grid::XRegularRG, order::Val{4}) = topological_diffusion_order(topology(grid, 1), order)
@inline y_diffusion_order(grid::YRegularRG, order::Val{4}) = topological_diffusion_order(topology(grid, 2), order)
@inline z_diffusion_order(grid::ZRegularRG, order::Val{4}) = topological_diffusion_order(topology(grid, 3), order)

# There is no stencil in `Flat` directions
@inline topological_diffusion_order(topology, order) = order
@inline topological_diffusion_order(::Type{Flat}, order) = Val(2)

@inline x_diffusion_order(ibg::ImmersedBoundaryGrid, order::Val{4}) = x_diffusion_order(ibg.underlying_grid, order)
@inline y_diffusion_order(ibg::ImmersedBoundaryGrid, order::Val{4}) = y_diffusion_order(ibg.underlying_grid, order)
@inline z_diffusion_order(ibg::ImmersedBoundaryGrid, order::Val{4}) = z_diffusion_order(ibg.underlying_grid, order)

# Partial cells change the vertical spacing near the immersed boundary
const PartialCellIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:PartialCellVerticalBoundaries}
@inline z_diffusion_order(::PartialCellIBG, ::Val{4}) = Val(2)

#####
##### Fourth-order differences of fields "located" at φ's location.
#####
##### `δ⁴ξᶠ` differences a field centered in ξ to faces; `δ⁴ξᶜ` differences a field on faces in ξ to centers.
#####

@inline δ⁴xᶠ(i, j, k, φ) = @inbounds (15 * (φ[i, j, k] - φ[i-1, j, k]) - (φ[i+1, j, k] - φ[i-2, j, k])) / 12
@inline δ⁴xᶜ(i, j, k, φ) = @inbounds (15 * (φ[i+1, j, k] - φ[i, j, k]) - (φ[i+2, j, k] - φ[i-1, j, k])) / 12
@inline δ⁴yᶠ(i, j, k, φ) = @inbounds (15 * (φ[i, j, k] - φ[i, j-1, k]) - (φ[i, j+1, k] - φ[i, j-2, k])) / 12
@inline δ⁴yᶜ(i, j, k, φ) = @inbounds (15 * (φ[i, j+1, k] - φ[i, j, k]) - (φ[i, j+2, k] - φ[i, j-1, k])) / 12
@inline δ⁴zᶠ(i, j, k, φ) = @inbounds (15 * (φ[i, j, k] - φ[i, j, k-1]) - (φ[i, j, k+1] - φ[i, j, k-2])) / 12
@inline δ⁴zᶜ(i, j, k, φ) = @inbounds (15 * (φ[i, j, k+1] - φ[i, j, k]) - (φ[i, j, k+2] - φ[i, j, k-1])) / 12

# Whether all four nodes of the stencil are active. `inactive_node` is `true` in the halos of `Bounded`
# directions and within immersed boundaries, and identically `false` in `Periodic` directions.
# ℓx, ℓy, ℓz is the location of the differenced field.
@inline active_x_stencil(i₁, j, k, grid, ℓx, ℓy, ℓz) =
    !(inactive_node(i₁,   j, k, grid, ℓx, ℓy, ℓz) | inactive_node(i₁+1, j, k, grid, ℓx, ℓy, ℓz) |
      inactive_node(i₁+2, j, k, grid, ℓx, ℓy, ℓz) | inactive_node(i₁+3, j, k, grid, ℓx, ℓy, ℓz))

@inline active_y_stencil(i, j₁, k, grid, ℓx, ℓy, ℓz) =
    !(inactive_node(i, j₁,   k, grid, ℓx, ℓy, ℓz) | inactive_node(i, j₁+1, k, grid, ℓx, ℓy, ℓz) |
      inactive_node(i, j₁+2, k, grid, ℓx, ℓy, ℓz) | inactive_node(i, j₁+3, k, grid, ℓx, ℓy, ℓz))

@inline active_z_stencil(i, j, k₁, grid, ℓx, ℓy, ℓz) =
    !(inactive_node(i, j, k₁,   grid, ℓx, ℓy, ℓz) | inactive_node(i, j, k₁+1, grid, ℓx, ℓy, ℓz) |
      inactive_node(i, j, k₁+2, grid, ℓx, ℓy, ℓz) | inactive_node(i, j, k₁+3, grid, ℓx, ℓy, ℓz))

#####
##### Derivatives used by diffusive fluxes: `diffusive_∂ξᴸ(i, j, k, grid, order, φ)` differentiates φ in ξ
##### to the location ᴸ, with second-order fallback.
#####

# (derivative name, direction, second-order derivative, fourth-order difference, stencil offset, location of φ)
fourth_order_derivatives = (
    # tracers
    (:diffusive_∂xᶠᶜᶜ, :x, :∂xᵣᶠᶜᶜ, :δ⁴xᶠ, -2, (:c, :c, :c)),
    (:diffusive_∂yᶜᶠᶜ, :y, :∂yᵣᶜᶠᶜ, :δ⁴yᶠ, -2, (:c, :c, :c)),
    (:diffusive_∂zᶜᶜᶠ, :z, :∂zᶜᶜᶠ,  :δ⁴zᶠ, -2, (:c, :c, :c)),
    # u
    (:diffusive_∂xᶜᶜᶜ, :x, :∂xᵣᶜᶜᶜ, :δ⁴xᶜ, -1, (:f, :c, :c)),
    (:diffusive_∂y_uᶠᶠᶜ, :y, :∂yᵣᶠᶠᶜ, :δ⁴yᶠ, -2, (:f, :c, :c)),
    (:diffusive_∂z_uᶠᶜᶠ, :z, :∂zᶠᶜᶠ,  :δ⁴zᶠ, -2, (:f, :c, :c)),
    # v
    (:diffusive_∂x_vᶠᶠᶜ, :x, :∂xᵣᶠᶠᶜ, :δ⁴xᶠ, -2, (:c, :f, :c)),
    (:diffusive_∂yᶜᶜᶜ, :y, :∂yᵣᶜᶜᶜ, :δ⁴yᶜ, -1, (:c, :f, :c)),
    (:diffusive_∂z_vᶜᶠᶠ, :z, :∂zᶜᶠᶠ,  :δ⁴zᶠ, -2, (:c, :f, :c)),
    # w
    (:diffusive_∂x_wᶠᶜᶠ, :x, :∂xᵣᶠᶜᶠ, :δ⁴xᶠ, -2, (:c, :c, :f)),
    (:diffusive_∂y_wᶜᶠᶠ, :y, :∂yᵣᶜᶠᶠ, :δ⁴yᶠ, -2, (:c, :c, :f)),
    (:diffusive_∂zᶜᶜᶜ, :z, :∂zᶜᶜᶜ,  :δ⁴zᶜ, -1, (:c, :c, :f)),
)

for (∂, ξ, ∂², δ⁴, offset, (ℓx, ℓy, ℓz)) in fourth_order_derivatives
    order_ξ = Symbol(ξ, :_diffusion_order)
    active_stencil = Symbol(:active_, ξ, :_stencil)
    Δξ = Symbol(:Δ, ξ, :ᶜᶜᶜ)
    shifted = ξ == :x ? :(i + $offset, j, k) : ξ == :y ? :(i, j + $offset, k) : :(i, j, k + $offset)

    @eval begin
        @inline $∂(i, j, k, grid, order, φ) = $∂(i, j, k, grid, $order_ξ(grid, order), φ)
        @inline $∂(i, j, k, grid, ::Val{2}, φ) = $∂²(i, j, k, grid, φ)

        @inline function $∂(i, j, k, grid, ::Val{4}, φ)
            ∂⁴φ = $δ⁴(i, j, k, φ) / $Δξ(i, j, k, grid)
            ∂²φ = $∂²(i, j, k, grid, φ)
            return ifelse($active_stencil($(shifted.args...), grid, $ℓx, $ℓy, $ℓz), ∂⁴φ, ∂²φ)
        end
    end
end

#####
##### Viscous "stresses" τᵢⱼ such that the viscous flux of uᵢ in direction j is -ν τᵢⱼ.
#####
##### With second-order diffusion, τᵢⱼ = 2Σᵢⱼ = ∂ⱼuᵢ + ∂ᵢuⱼ. With fourth-order diffusion we compute
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
