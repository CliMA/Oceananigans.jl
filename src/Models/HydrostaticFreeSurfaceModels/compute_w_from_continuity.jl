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

# extend w kernel to compute also the boundaries
@inline w_kernel_size(grid)    = size(grid)[[1, 2]] .+ halo_size(grid)[[1, 2]] .- 2
@inline w_kernel_offsets(grid) = - halo_size(grid)[[1, 2]] .+ 1

@kernel function _compute_w_from_continuity!(U, offs, grid)
    i, j = @index(Global, NTuple)

    i′ = i + offs[1] 
    j′ = j + offs[2] 

    U.w[i′, j′, 1] = 0
    @unroll for k in 2:grid.Nz+1
        @inbounds U.w[i′, j′, k] = U.w[i′, j′, k-1] - Δzᶜᶜᶜ(i′, j′, k-1, grid) * div_xyᶜᶜᶜ(i′, j′, k-1, grid, U.u, U.v)
    end
end
