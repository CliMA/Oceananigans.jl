"Returns the time-scale for advection on a regular grid across a single grid cell 
 for ShallowWaterModel."

import Oceananigans.Utils: cell_advection_timescale

function shallow_water_cell_advection_timescale(uh, vh, h, grid, formulation)
    
    Δxmin = minimum(grid.Δxᶜᵃᵃ)
    Δymin = minimum(grid.Δyᵃᶜᵃ)
    
    uhmax = maximum(abs, uh)
    vhmax = maximum(abs, vh)
    hmin  = minimum(abs,  h)

    return min(Δxmin / uhmax, Δymin / vhmax) * hmin
end

function shallow_water_cell_advection_timescale(u, v, h, grid, ::VectorInvariantFormulation)
    
    Δxmin = minimum(grid.Δxᶜᵃᵃ)
    Δymin = minimum(grid.Δyᵃᶜᵃ)
    
    umax = maximum(abs, u)
    vmax = maximum(abs, v)

    return min(Δxmin / umax, Δymin / vmax)
end

cell_advection_timescale(model::ShallowWaterModel) =
    shallow_water_cell_advection_timescale(
        model.solution[1].data.parent, 
        model.solution[2].data.parent,
        model.solution.h.data.parent,
        model.grid, model.formulation
        )
