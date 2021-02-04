"Returns the time-scale for advection on a regular grid across a single grid cell 
 for ShallowWaterModel."

import Oceananigans.Utils: cell_advection_timescale

cell_advection_timescale(model::ShallowWaterModel) = shallow_water_cell_advection_timescale(model.solution, model.grid)

function shallow_water_cell_advection_timescale(solution::ConservativeSolutionFields, grid)
    uhmax = maximum(abs, solution.uh.data.parent)
    vhmax = maximum(abs, solution.vh.data.parent)
    hmin  = minimum(abs, solution.h.data.parent)

    return min(grid.Δx / uhmax, grid.Δy / vhmax) * hmin
end

function shallow_water_cell_advection_timescale(solution::PrimitiveSolutionLinearizedHeightFields, grid)
    umax = maximum(abs, solution.u.data.parent)
    vmax = maximum(abs, solution.v.data.parent)

    return min(grid.Δx / umax, grid.Δy / vmax)
end
