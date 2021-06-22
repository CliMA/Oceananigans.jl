using Oceananigans.Grids: topology, min_Δx, min_Δy, min_Δz

"Returns the time-scale for advection on a regular grid across a single grid cell."
function cell_advection_timescale(u, v, w, grid)

    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    Δx = min_Δx(grid)
    Δy = min_Δy(grid)
    Δz = min_Δz(grid)

    return min(Δx/umax, Δy/vmax, Δz/wmax)
end

cell_advection_timescale(model) =
    cell_advection_timescale(model.velocities.u.data.parent,
                             model.velocities.v.data.parent,
                             model.velocities.w.data.parent,
                             model.grid)
