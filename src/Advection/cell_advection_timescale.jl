using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Operators: Δz⁻¹ᶜᶜᶠ

"""
    cell_advection_timescale(grid, velocities)

Return the advection timescale for `grid` with `velocities`. The advection timescale
is the minimum over all `i, j, k` in the `grid` of

```
  1 / (|u(i, j, k)| / Δxᶠᶜᶜ(i, j, k) + |v(i, j, k)| / Δyᶜᶠᶜ(i, j, k) + |w(i, j, k)| / Δzᶜᶜᶠ(i, j, k))
```
"""
function cell_advection_timescale(grid, velocities)
    u = u_velocity(velocities)
    v = v_velocity(velocities)
    w = w_velocity(velocities)
    τ = KernelFunctionOperation{Center, Center, Center}(cell_advection_timescaleᶜᶜᶜ, grid, u, v, w)
    return minimum(τ)
end

@inline function transport_cell_advection_timescaleᶜᶜᶜ(i, j, k, grid::SphericalShellGrid, u, v, w)
    inverse_timescale_x = abs(@inbounds u[i, j, k]) * Oceananigans.Operators.V⁻¹ᶜᶜᶜ(i, j, k, grid)
    inverse_timescale_y = abs(@inbounds v[i, j, k]) * Oceananigans.Operators.V⁻¹ᶜᶜᶜ(i, j, k, grid)
    inverse_timescale_z = abs(@inbounds w[i, j, k]) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid)

    return one(grid) / (inverse_timescale_x + inverse_timescale_y + inverse_timescale_z)
end

function cell_advection_timescale(grid::SphericalShellGrid, velocities)
    transport_velocities = spherical_shell_volume_flux_velocities(grid, velocities)
    τ = KernelFunctionOperation{Center, Center, Center}(transport_cell_advection_timescaleᶜᶜᶜ,
                                                        grid,
                                                        u_velocity(transport_velocities),
                                                        v_velocity(transport_velocities),
                                                        w_velocity(transport_velocities))
    return minimum(τ)
end

@inline _inverse_timescale(i, j, k, Δ⁻¹, U, topo) = @inbounds abs(U[i, j, k]) * Δ⁻¹
@inline _inverse_timescale(i, j, k, Δ⁻¹, U, topo::Flat) = 0

@inline function cell_advection_timescaleᶜᶜᶜ(i, j, k, grid::AbstractGrid{FT, TX, TY, TZ}, u, v, w) where {FT, TX, TY, TZ}
    Δx⁻¹ = Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    Δy⁻¹ = Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    Δz⁻¹ = Δz⁻¹ᶜᶜᶠ(i, j, k, grid)

    inverse_timescale_x = _inverse_timescale(i, j, k, Δx⁻¹, u, TX())
    inverse_timescale_y = _inverse_timescale(i, j, k, Δy⁻¹, v, TY())
    inverse_timescale_z = _inverse_timescale(i, j, k, Δz⁻¹, w, TZ())

    inverse_timescale = inverse_timescale_x + inverse_timescale_y + inverse_timescale_z

    return 1 / inverse_timescale
end
