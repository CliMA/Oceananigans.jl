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


# Since the derivative of the moving grid is:
#
#            δx(Δy U) + δy(Δx V)       ∇ ⋅ U
# ∂t_e₃ = - --------------------- = - --------
#                   Az ⋅ H               H    
#
# The discrete divergence is calculated as:
#
#  wᵏ⁺¹ - wᵏ      δx(Ax u) + δy(Ay v)     Δr ∂t_e₃
# ---------- = - --------------------- - ----------
#     Δz                  V                  Δz
#
# This makes sure that if we sum up till the top of the domain, we get
#
#                ∇ ⋅ U
#  wᴺᶻ⁺¹ = w⁰ - ------- - ∂t_e₃ ≈ 0 (if w⁰ == 0)
#                  H   
# 
@kernel function _compute_w_from_continuity!(U, grid)
    i, j = @index(Global, NTuple)

    @inbounds U.w[i, j, 1] = 0
    for k in 2:grid.Nz+1
        δh_u = flux_div_xyᶜᶜᶜ(i, j, k-1, grid, U.u, U.v) / Azᶜᶜᶜ(i, j, k-1, grid) 
        ∂te₃ = Δrᶜᶜᶜ(i, j, k-1, grid) * ∂t_e₃(i, j, k-1, grid)

        immersed = immersed_cell(i, j, k-1, grid)
        Δw       = δh_u + ifelse(immersed, 0, ∂te₃) # We do not account for grid changes in immersed cells

        @inbounds U.w[i, j, k] = U.w[i, j, k-1] - Δw
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
