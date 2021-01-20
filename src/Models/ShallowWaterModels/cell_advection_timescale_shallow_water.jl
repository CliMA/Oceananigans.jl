"Returns the time-scale for advection on a regular grid across a single grid cell 
 for ShallowWaterModel."
function cell_advection_timescale(uh, vh, h, grid)
    uhmax = maximum(abs, uh)
    vhmax = maximum(abs, vh)
     hmin = minimum(abs,  h)

     umax = uhmax/hmin
     vmax = vhmax/hmin

    Δx = grid.Δx
    Δy = grid.Δy

    return min(Δx/umax, Δy/vmax)
end

cell_advection_timescale(model::ShallowWaterModel) =
    cell_advection_timescale(model.solution.uh.data.parent,
                             model.solution.vh.data.parent,
                             model.solution.h.data.parent,
                             model.grid)
