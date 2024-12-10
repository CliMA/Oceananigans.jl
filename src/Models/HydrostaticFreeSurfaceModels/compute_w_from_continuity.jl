using Oceananigans.Architectures: device
using Oceananigans.Models: surface_kernel_parameters
using Oceananigans.Grids: halo_size, topology
using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans.Operators: div_xyᶜᶜᶜ, Δzᶜᶜᶜ

"""
    compute_w_from_continuity!(model)

Compute the vertical velocity ``w`` by integrating the continuity equation from the bottom upwards:

```
w^{n+1} = -∫ [∂/∂x (u^{n+1}) + ∂/∂y (v^{n+1})] dz
```
"""
compute_w_from_continuity!(model; kwargs...) =
    compute_w_from_continuity!(model.velocities, model.architecture, model.grid; kwargs...)

compute_w_from_continuity!(velocities, arch, grid; parameters = surface_kernel_parameters(arch, grid)) = 
    launch!(arch, grid, parameters, _compute_w_from_continuity!, velocities, grid)

@kernel function _compute_w_from_continuity!(U, grid)
    i, j = @index(Global, NTuple)

    @inbounds U.w[i, j, 1] = 0
    for k in 2:grid.Nz+1
        @inbounds U.w[i, j, k] = U.w[i, j, k-1] - Δzᶜᶜᶜ(i, j, k-1, grid) * div_xyᶜᶜᶜ(i, j, k-1, grid, U.u, U.v)
    end
end