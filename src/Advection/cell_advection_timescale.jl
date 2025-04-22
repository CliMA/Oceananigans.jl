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

@inline _inverse_timescale(i, j, k, Δ, U, topo) = @inbounds abs(U[i, j, k]) / Δ
@inline _inverse_timescale(i, j, k, Δ, U, topo::Flat) = 0

@inline function cell_advection_timescaleᶜᶜᶜ(i, j, k, grid::AbstractGrid{FT, TX, TY, TZ}, u, v, w) where {FT, TX, TY, TZ}
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    Δz = Δzᶜᶜᶠ(i, j, k, grid)

    inverse_timescale_x = _inverse_timescale(i, j, k, Δx, u, TX())
    inverse_timescale_y = _inverse_timescale(i, j, k, Δy, v, TY())
    inverse_timescale_z = _inverse_timescale(i, j, k, Δz, w, TZ())

    inverse_timescale = inverse_timescale_x + inverse_timescale_y + inverse_timescale_z

    return 1 / inverse_timescale
end
