"Returns the time-scale for advection on a regular grid across a single grid cell."

function cell_advection_timescale(model::ShallowWaterModel)
    hmin = minimum(abs, model.solution.h)
    umax = maximum(abs, model.solution.uh) / hmin
    vmax = maximum(abs, model.solution.vh) / hmin

    Δx = model.grid.Δx
    Δy = model.grid.Δy

    return min(Δx/umax, Δy/vmax)
end

cell_advection_timescale(model) =
    cell_advection_timescale(model.solution.uh.data.parent,
                             model.solution.vh.data.parent,
                             model.grid)
