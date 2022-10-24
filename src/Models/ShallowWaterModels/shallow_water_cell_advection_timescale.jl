"Returns the time-scale for advection on a regular grid across a single grid cell 
 for ShallowWaterModel."

import Oceananigans.Utils: cell_advection_timescale

function shallow_water_cell_advection_timescale(::ConservativeFormulation, uh, vh, h, grid::RectilinearGrid)
    
    Δxmin = minimum(grid.Δxᶜᵃᵃ)
    Δymin = minimum(grid.Δyᵃᶜᵃ)
    
    uhmax = maximum(abs, uh)
    vhmax = maximum(abs, vh)
    hmin  = minimum(abs,  h)

    return min(Δxmin / uhmax, Δymin / vhmax) * hmin
end

function shallow_water_cell_advection_timescale(::VectorInvariantFormulation, u, v, h, grid::RectilinearGrid)
    
    Δxmin = minimum(grid.Δxᶜᵃᵃ)
    Δymin = minimum(grid.Δyᵃᶜᵃ)
    
    umax = maximum(abs, u)
    vmax = maximum(abs, v)

    return min(Δxmin / umax, Δymin / vmax)
end

cell_advection_timescale(model::ShallowWaterModel) = shallow_water_cell_advection_timescale(model.formulation,
                                                                                            model.solution[1],
                                                                                            model.solution[2],
                                                                                            model.solution.h,
                                                                                            model.grid)

