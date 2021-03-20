"Returns the time-scale for advection on a regular grid across a single grid cell 
 for ShallowWaterModel."

using Oceananigans.Grids: Flat

import Oceananigans.Utils: cell_advection_timescale

@inline x_cell_advection_timescale(grid::RegularRectilinearGrid, uhmax, hmin) = grid.Δx / uhmax * hmin
@inline y_cell_advection_timescale(grid::RegularRectilinearGrid, vhmax, hmin) = grid.Δy / vhmax * hmin

x_cell_advection_timescale(grid::RegularRectilinearGrid{FT, <:Flat, TY, TZ}, uhmax, hmin) where {FT, TY, TZ} = Inf
y_cell_advection_timescale(grid::RegularRectilinearGrid{FT, TX, <:Flat, TZ}, vhmax, hmin) where {FT, TX, TZ} = Inf

function shallow_water_cell_advection_timescale(uh, vh, h, grid)
    uhmax = maximum(abs, uh)
    vhmax = maximum(abs, vh)
    hmin  = minimum(abs,  h)

    return min(
        x_cell_advection_timescale(grid, uhmax, hmin),
        y_cell_advection_timescale(grid, vhmax, hmin)
       )
end

cell_advection_timescale(model::ShallowWaterModel) =
    shallow_water_cell_advection_timescale(
        model.solution.uh.data.parent, 
        model.solution.vh.data.parent,
        model.solution.h.data.parent,
        model.grid
        )
