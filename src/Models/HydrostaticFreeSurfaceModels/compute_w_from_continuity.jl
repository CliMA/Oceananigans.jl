using Oceananigans.Architectures: device
using Oceananigans.Grids: halo_size, topology
using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, div_xyᶜᶜᶜ, Δzᶜᶜᶜ

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

@inline scaled_velocity_divergenceᶜᶜᶜ(i, j, k, grid, u, v) = flux_div_xyᶜᶜᶜ(i, j, k, grid, u, v) / Azᶜᶜᶜ(i, j, k, grid)

@kernel function _compute_w_from_continuity!(U, grid)
    i, j = @index(Global, NTuple)

    @inbounds U.w[i, j, 1] = 0
    for k in 2:grid.Nz+1
        @inbounds U.w[i, j, k] = U.w[i, j, k-1] - 
                                ( scaled_velocity_divergenceᶜᶜᶜ(i, j, k-1, grid, U.u, U.v) +
                                 Δrᶜᶜᶜ(i, j, k-1, grid) *  ∂t_s_grid(i, j, k-1, grid) )
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
