using Oceananigans.AbstractOperations: KernelFunctionOperation

"""
    cell_advection_timescale(grid, velocities)

Return the advection timescale for `grid` with `velocities`. The advection timescale
is the minimum over all `i, j, k` in the `grid` of

```
  1 / (|u(i, j, k)| / Δxᶠᶜᶜ(i, j, k) + |v(i, j, k)| / Δyᶜᶠᶜ(i, j, k) + |w(i, j, k)| / Δzᶜᶜᶠ(i, j, k))
```
"""
function cell_advection_timescale(grid, velocities)
    u, v, w = velocities
    τ = KernelFunctionOperation{Center, Center, Center}(cell_advection_timescaleᶜᶜᶜ, grid, u, v, w)
    return minimum(τ)
end

@inline function cell_advection_timescaleᶜᶜᶜ(i, j, k, grid, u, v, w)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    Δz = Δzᶜᶜᶠ(i, j, k, grid)

    inverse_timescale = @inbounds abs(u[i, j, k]) / Δx + abs(v[i, j, k]) / Δy + abs(w[i, j, k]) / Δz
     
    return 1 / inverse_timescale
end
