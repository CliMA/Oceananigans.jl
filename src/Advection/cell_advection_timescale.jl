using Oceananigans.AbstractOperations: KernelFunctionOperation
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


    ω = @inbounds abs(u[i, j, k]) / Δx + abs(v[i, j, k]) / Δy + abs(w[i, j, k]) / Δz
    return 1 / ω
end
