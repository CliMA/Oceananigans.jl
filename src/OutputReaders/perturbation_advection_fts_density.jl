using Oceananigans.Grids: node
using Oceananigans.Fields: instantiated_location
using Oceananigans.Units: Time

import Oceananigans.BoundaryConditions: to_intensive, to_extensive

# FTS-density overloads of `PerturbationAdvection`'s `to_intensive` / `to_extensive`.
# Spatial + temporal interpolation through the FTS machinery, so the density FTS
# may live on a different grid (and at a different location) than the simulation
# variable being converted. This is the regional-hindcast case where
# `ρ_boundary(x, y, z, t)` is diagnosed from reanalysis thermodynamics.
@inline _pa_fts_density_value(ρ_fts, i, j, k, grid, loc, time) =
    interpolate(node(i, j, k, grid, loc...), Time(time), ρ_fts,
                instantiated_location(ρ_fts), ρ_fts.grid)

@inline to_intensive(ρ::FlavorOfFTS, ψ, i, j, k, grid, loc, clock) =
    @inbounds ψ[i, j, k] / _pa_fts_density_value(ρ, i, j, k, grid, loc, clock.time)

@inline to_extensive(ρ::FlavorOfFTS, ψ_value, i, j, k, grid, loc, clock) =
    _pa_fts_density_value(ρ, i, j, k, grid, loc, clock.time) * ψ_value
