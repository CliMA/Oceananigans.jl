"Returns the time-scale for advection on a regular grid across a single grid cell
 for ShallowWaterModel."

import Oceananigans.Advection: cell_advection_timescale
using Oceananigans.Grids: SphericalShellGrid
using Oceananigans.Operators: Azᶜᶜᶜ

function cell_advection_timescale(model::ShallowWaterModel)
    refresh_shallow_water_auxiliary_state!(model)
    u, v, _ = shallow_water_velocities(model)
    τ = KernelFunctionOperation{Center, Center, Nothing}(shallow_water_cell_advection_timescaleᶜᶜᵃ, model.grid, u, v)
    return minimum(τ)
end

function cell_advection_timescale(model::ShallowWaterModel{<:SphericalShellGrid})
    refresh_shallow_water_auxiliary_state!(model)
    u, v, _ = shallow_water_velocities(model)
    transport_velocities =
        Oceananigans.Advection.spherical_shell_horizontal_volume_flux_velocities(model.grid, (u, v))
    τ = KernelFunctionOperation{Center, Center, Nothing}(shallow_water_cell_advection_timescaleᶜᶜᵃ,
                                                         model.grid,
                                                         Oceananigans.Advection.u_velocity(transport_velocities),
                                                         Oceananigans.Advection.v_velocity(transport_velocities))
    return minimum(τ)
end

@inline function shallow_water_cell_advection_timescaleᶜᶜᵃ(i, j, k, grid, u, v)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    return @inbounds min(Δx / abs(u[i, j, k]), Δy / abs(v[i, j, k]))
end

@inline function shallow_water_cell_advection_timescaleᶜᶜᵃ(i, j, k, grid::SphericalShellGrid, u, v)
    Az = Azᶜᶜᶜ(i, j, k, grid)

    return @inbounds min(Az / abs(u[i, j, k]), Az / abs(v[i, j, k]))
end
