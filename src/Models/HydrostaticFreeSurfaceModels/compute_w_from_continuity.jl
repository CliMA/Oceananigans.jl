using Oceananigans.Grids: halo_size, topology
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, Az⁻¹ᶜᶜᶜ, Δrᶜᶜᶜ, ∂t_σ
using Oceananigans.ImmersedBoundaries: immersed_cell
using Oceananigans.Models: surface_kernel_parameters

"""
    update_vertical_velocities!(velocities, grid, model; parameters=surface_kernel_parameters(grid))

Update the vertical velocity field `w` and grid vertical velocity (for z-star coordinates).

This function:
1. Updates the grid vertical velocity `∂t_σ` for mutable grids (z-star coordinates)
2. Computes `w` from the continuity equation by vertical integration

For static grids, only step 2 is performed. The `velocities` argument can be either
`model.velocities` (for momentum) or `model.transport_velocities` (for tracer advection).
"""
function update_vertical_velocities!(velocities, grid, model; parameters = surface_kernel_parameters(grid))
    update_grid_vertical_velocity!(velocities, model, grid, model.vertical_coordinate; parameters)
    compute_w_from_continuity!(velocities, grid; parameters)
    return nothing
end

"""
    update_grid_vertical_velocity!(velocities, model, grid, vertical_coordinate; kw...)

Update the time derivative of the grid stretching factor `∂t_σ` for mutable vertical coordinates.

Fallback method that does nothing (for static grids). Extended for `ZStarCoordinate` to compute
`∂t_σ = - ∇·U / H` where `U` is either the barotropic velocities or the barotropic transport.
"""
update_grid_vertical_velocity!(velocities, model, grid, ztype; kw...) = nothing

"""
    compute_w_from_continuity!(model; kwargs...)
    compute_w_from_continuity!(velocities, grid; parameters=surface_kernel_parameters(grid))

Compute the vertical velocity `w` by integrating the continuity equation from the bottom upwards:

```math
w^{n+1} = -\\int [\\partial u / \\partial x + \\partial v / \\partial y + \\partial_t \\sigma] dz
```

where `∂t_σ` is the time derivative of the grid stretching factor (zero for static grids).

The first method dispatches on `model.velocities` and `model.grid`. The second method
allows computing `w` for arbitrary velocity fields (e.g., `model.transport_velocities`).
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
