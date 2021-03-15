"Returns the time-scale for advection on a regular grid across a single grid cell."
function cell_advection_timescale(u, v, w, grid)
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    Δx = grid.Δx
    Δy = grid.Δy
    Δz = grid.Δz

    return min(Δx/umax, Δy/vmax, Δz/wmax)
end


function cell_advection_timescale(u, v, w, grid::VerticallyStretchedRectilinearGrid)
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    Δx = grid.Δx
    Δy = grid.Δy
    Δz = minimum(grid.Δzᵃᵃᶜ[1:grid.Nz])

    return min(Δx/umax, Δy/vmax, Δz/wmax)
end



cell_advection_timescale(model) =
    cell_advection_timescale(model.velocities.u.data.parent,
                             model.velocities.v.data.parent,
                             model.velocities.w.data.parent,
                             model.grid)
