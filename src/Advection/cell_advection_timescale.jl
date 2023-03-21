using Oceananigans.Grids: topology, min_Δx, min_Δy, min_Δz

function cell_advection_timescale(grid, velocities)
    u, v, w = velocities
    τ = KernelFunctionOperation{Center, Center, Center}(cell_advection_timescaleᶜᶜᶜ, grid, u, v, w)
    return minimum(τ)
end

@inline function cell_advection_timescaleᶜᶜᶜ(i, j, k, grid, u, v, w)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    Δz = Δzᶜᶜᶠ(i, j, k, grid)

    return @inbounds min(Δx / abs(u[i, j, k]),
                         Δy / abs(v[i, j, k]),
                         Δz / abs(w[i, j, k])) 
end

