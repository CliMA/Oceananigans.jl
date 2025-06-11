using Oceananigans.Architectures: device
using Oceananigans.Grids: halo_size, topology
using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, div_xyᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.ImmersedBoundaries: immersed_cell

"""
    compute_w_from_continuity!(model)

Compute the vertical velocity ``w`` by integrating the continuity equation from the bottom upwards:

```
w^{n+1} = -∫ [∂/∂x (u^{n+1}) + ∂/∂y (v^{n+1})] dz
```
"""
compute_w_from_continuity!(model; kwargs...) =
    compute_w_from_continuity!(model.velocities, model.architecture, model.grid; kwargs...)

compute_w_from_continuity!(velocities, arch, grid; parameters = w_kernel_parameters(grid)) =
    launch!(arch, grid, parameters, _compute_w_from_continuity!, velocities, grid)

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

        # We do not account for grid changes in immersed cells
        not_immersed = !immersed_cell(i, j, k-1, grid)
        w̃ = Δrᶜᶜᶜ(i, j, k-1, grid) * ∂t_σ(i, j, k-1, grid) * not_immersed

        wᵏ -= (δ + w̃)
        @inbounds w[i, j, k] = wᵏ
    end
end

#####
##### Size and offsets for the w kernel
#####

# extend w kernel to compute also the boundaries
# If Flat, do not calculate on halos!
@inline function w_kernel_parameters(grid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)

    ii = ifelse(Tx == Flat, 1:Nx, -Hx+2:Nx+Hx-1)
    jj = ifelse(Ty == Flat, 1:Ny, -Hy+2:Ny+Hy-1)

    return KernelParameters(ii, jj)
end
