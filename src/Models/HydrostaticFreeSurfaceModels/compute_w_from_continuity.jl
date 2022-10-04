using Oceananigans.Architectures: device, device_event
using Oceananigans.Operators: div_xyᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.TimeSteppers: tendency_kernel_size, tendency_kernel_offset

"""
    compute_w_from_continuity!(model)

Compute the vertical velocity ``w`` by integrating the continuity equation from the bottom upwards:

```
w^{n+1} = -∫ [∂/∂x (u^{n+1}) + ∂/∂y (v^{n+1})] dz
```
"""
compute_w_from_continuity!(model; kwargs...) = compute_w_from_continuity!(model.velocities, model.architecture, model.grid; kwargs...)

function compute_w_from_continuity!(velocities, arch, grid; region_to_compute = :interior, dependencies = device_event(arch))

    kernel_size    = tendency_kernel_size(grid, Val(region_to_compute))[[1, 2]]
    kernel_offsets = tendency_kernel_offset(grid, Val(region_to_compute))[[1, 2]]

    event = launch!(arch,
                    grid,
                    kernel_size,
                    _compute_w_from_continuity!,
                    velocities,
                    kernel_offsets,
                    grid;
                    dependencies)

    return event
end

@kernel function _compute_w_from_continuity!(U, offsets, grid)
    i, j = @index(Global, NTuple)
    i′ = i + offsets[1]
    j′ = j + offsets[2]
    U.w[i′, j′, 1] = 0
    @unroll for k in 2:grid.Nz+1
        @inbounds U.w[i′, j′, k] = U.w[i′, j′, k-1] - Δzᶜᶜᶜ(i′, j′, k-1, grid) * div_xyᶜᶜᶜ(i′, j′, k-1, grid, U.u, U.v)
    end
end
