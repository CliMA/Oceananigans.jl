using Oceananigans.Architectures: device, device_event
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.AdvectionDivergence: div_xyᶜᶜᶜ, Δzᶜᶜᶜ

"""
    compute_w_from_continuity!(model)

Compute the vertical velocity ``w`` by integrating the continuity equation from the bottom upwards:

```
w^{n+1} = -∫ [∂/∂x (u^{n+1}) + ∂/∂y (v^{n+1})] dz
```
"""
compute_w_from_continuity!(model) = compute_w_from_continuity!(model.velocities, model.architecture, model.grid)

function compute_w_from_continuity!(velocities, arch, grid)

    event = launch!(arch, grid, :xy,
                    _compute_w_from_continuity!, velocities, grid,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@kernel function _compute_w_from_continuity!(U, grid)
    i, j = @index(Global, NTuple)
    U.w[i, j, 1] = 0
    @unroll for k in 2:grid.Nz+1
        @inbounds U.w[i, j, k] = U.w[i, j, k-1] - Δzᶜᶜᶜ(i, j, k-1, grid) * div_xyᶜᶜᶜ(i, j, k-1, grid, U.u, U.v)
    end
end

