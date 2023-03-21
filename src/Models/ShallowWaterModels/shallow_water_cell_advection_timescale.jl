"Returns the time-scale for advection on a regular grid across a single grid cell 
 for ShallowWaterModel."

import Oceananigans.Advection: cell_advection_timescale

function cell_advection_timescale(model::ShallowWaterModel)
    u, v, _ = shallow_water_velocities(model)
    τ = KernelFunctionOperation{Center, Center, Nothing}(shallow_water_cell_advection_timescaleᶜᶜᵃ, grid, u, v)
    return minimum(τ)
end

@inline function shallow_water_cell_advection_timescaleᶜᶜᵃ(i, j, k, grid, u, v)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    return @inbounds min(Δx / abs(u[i, j, k]), Δy / abs(v[i, j, k]))
end

