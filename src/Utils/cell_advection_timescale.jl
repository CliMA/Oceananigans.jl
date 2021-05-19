using Oceananigans.Grids: topology

minimum_grid_spacing(Δx, TX          ) = CUDA.@allowscalar minimum(Δx)
minimum_grid_spacing(Δx, ::Type{Flat}) = Inf

"Returns the time-scale for advection on a regular grid across a single grid cell."
function cell_advection_timescale(u, v, w, grid::RegularRectilinearGrid)

    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    topo = topology(grid)

    Δx = minimum_grid_spacing(grid.Δx, topo[1])
    Δy = minimum_grid_spacing(grid.Δy, topo[2])
    Δz = minimum_grid_spacing(grid.Δz, topo[3])

    return min(Δx/umax, Δy/vmax, Δz/wmax)
end

function cell_advection_timescale(u, v, w, grid::VerticallyStretchedRectilinearGrid)

    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    topo = topology(grid)

    Δx = minimum_grid_spacing(grid.Δx,    topo[1])
    Δy = minimum_grid_spacing(grid.Δy,    topo[2])
    Δz = minimum_grid_spacing(grid.Δzᵃᵃᶠ, topo[3])

    return min(Δx/umax, Δy/vmax, Δz/wmax)
end

cell_advection_timescale(model) =
    cell_advection_timescale(model.velocities.u.data.parent,
                             model.velocities.v.data.parent,
                             model.velocities.w.data.parent,
                             model.grid)
