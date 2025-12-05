using Oceananigans.Grids: halo_size, topology
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, Az⁻¹ᶜᶜᶜ, Δrᶜᶜᶜ, ∂t_σ
using Oceananigans.ImmersedBoundaries: immersed_cell
using Oceananigans.Models: surface_kernel_parameters

function update_vertical_velocities!(velocities, grid, model; parameters = surface_kernel_parameters(grid))
    update_grid_vertical_velocity!(velocities, model, grid, model.vertical_coordinate; parameters)
    compute_w_from_continuity!(velocities, grid; parameters)
    return nothing
end

# A Fallback to be extended for specific ztypes and grid types
update_grid_vertical_velocity!(velocities, model, grid, ztype; kw...) = nothing

"""
    compute_w_from_continuity!(model)

Compute the vertical velocity ``w`` by integrating the continuity equation from the bottom upwards:

```
w^{n+1} = -∫ [∂/∂x (u^{n+1}) + ∂/∂y (v^{n+1})] dz
```
"""
compute_w_from_continuity!(model; kwargs...) =
    compute_w_from_continuity!(model.velocities, model.grid; kwargs...)

compute_w_from_continuity!(velocities, grid; parameters = surface_kernel_parameters(grid)) =
    launch!(architecture(grid), grid, parameters, _compute_w_from_continuity!, velocities, grid)

# If the grid is following the free surface, then the derivative of the moving grid is:
#
#            δx(Δy U) + δy(Δx V)       ∇ ⋅ U
# ∂t_σ = - --------------------- = - --------
#                   Az ⋅ H               H
#
# The discrete divergence is then calculated as:
#
#  wᵏ⁺¹ - wᵏ      δx(Ax u) + δy(Ay v)     Δr ∂t_σ
# ---------- = - --------------------- - ----------
#     Δz                 vol                 Δz
#
# This makes sure that summing up till the top of the domain, results in:
#
#                ∇ ⋅ U
#  wᴺᶻ⁺¹ = w⁰ - ------- - ∂t_σ ≈ 0 (if w⁰ == 0)
#                  H
#
# If the grid is static, then ∂t_σ = 0 and the moving grid contribution is equal to zero
@kernel function _compute_w_from_continuity!(U, grid)
    i, j = @index(Global, NTuple)

    u, v, w = U
    wᵏ = zero(eltype(w))
    @inbounds w[i, j, 1] = wᵏ

    Nz = size(grid, 3)
    for k in 2:Nz+1
        δ = flux_div_xyᶜᶜᶜ(i, j, k-1, grid, u, v) * Az⁻¹ᶜᶜᶜ(i, j, k-1, grid)
        w̃ = Δrᶜᶜᶜ(i, j, k-1, grid) * ∂t_σ(i, j, k-1, grid)

        # We do not account for grid changes in immersed cells
        immersed = immersed_cell(i, j, k-1, grid)
        w̃ = ifelse(immersed, zero(grid), w̃)

        wᵏ -= (δ + w̃)
        @inbounds w[i, j, k] = wᵏ
    end
end
