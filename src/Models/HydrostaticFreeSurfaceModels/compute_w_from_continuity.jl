using Oceananigans.Architectures: device
using Oceananigans.Operators: div_xyᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.Grids: halo_size

"""
    compute_w_from_continuity!(model)

Compute the vertical velocity ``w`` by integrating the continuity equation from the bottom upwards:

```
w^{n+1} = -∫ [∂/∂x (u^{n+1}) + ∂/∂y (v^{n+1})] dz
```
"""
compute_w_from_continuity!(model) = compute_w_from_continuity!(model.velocities, model.architecture, model.grid)

compute_w_from_continuity!(velocities, arch, grid; kernel_size = w_kernel_size(grid), kernel_offsets = w_kernel_offsets(grid)) = 
    launch!(arch, grid, kernel_size, _compute_w_from_continuity!, velocities, kernel_offsets, grid)

@kernel function _compute_w_from_continuity!(U, offs, grid)
    i, j = @index(Global, NTuple)

    i′ = i + offs[1] 
    j′ = j + offs[2] 

    U.w[i′, j′, 1] = 0
    @unroll for k in 2:grid.Nz+1
        @inbounds U.w[i′, j′, k] = U.w[i′, j′, k-1] - Δzᶜᶜᶜ(i′, j′, k-1, grid) * div_xyᶜᶜᶜ(i′, j′, k-1, grid, U.u, U.v)
    end
end

#####
##### Size and offsets for the w kernel
#####

# extend w kernel to compute also the boundaries
# If Flat, do not calculate on halos!

using Oceananigans.Operators: XFlatGrid, YFlatGrid
using Oceananigans.Grids: topology

@inline function w_kernel_size(grid) 
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    Tx, Ty, _ = topology(grid)

    Ax = Tx == Flat ? Nx : Nx + 2Hx - 2 
    Ay = Ty == Flat ? Ny : Ny + 2Hy - 2 

    return (Ax, Ay)
end

@inline function w_kernel_offsets(grid)
    Hx, Hy, _ = halo_size(grid)

    Tx, Ty, _ = topology(grid)

    Ax = Tx == Flat ? 0 : - Hx + 1 
    Ay = Ty == Flat ? 0 : - Hy + 1 

    return (Ax, Ay)
end

