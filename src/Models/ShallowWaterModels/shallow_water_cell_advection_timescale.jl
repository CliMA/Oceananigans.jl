"Returns the time-scale for advection on a regular grid across a single grid cell 
 for ShallowWaterModel."

import Oceananigans.Utils: cell_advection_timescale

function shallow_water_cell_advection_timescale(uh, vh, h, grid)
    
    Δxmin = minimum(grid.Δxᶜᵃᵃ)
    Δymin = minimum(grid.Δyᵃᶜᵃ)
    
    uhmax = maximum(abs, uh)
    vhmax = maximum(abs, vh)
    hmin  = minimum(abs,  h)

    return min(Δxmin / uhmax, Δymin / vhmax) * hmin
end

cell_advection_timescale(model::ShallowWaterModel) =
    shallow_water_cell_advection_timescale(
        model.solution.uh.data.parent, 
        model.solution.vh.data.parent,
        model.solution.h.data.parent,
        model.grid
        )
