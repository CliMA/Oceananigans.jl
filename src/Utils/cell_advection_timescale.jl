using Oceananigans.Grids: Flat

@inline x_cell_advection_timescale(grid::RegularCartesianGrid, umax) = grid.Δx / umax
@inline y_cell_advection_timescale(grid::RegularRectilinearGrid, vmax) = grid.Δy / vmax
@inline z_cell_advection_timescale(grid::RegularRectilinearGrid, wmax) = grid.Δz / wmax

x_cell_advection_timescale(grid::RegularRectilinearGrid{FT, <:Flat, TY, TZ}, umax) where {FT, TY, TZ} = Inf
y_cell_advection_timescale(grid::RegularRectilinearGrid{FT, TX, <:Flat, TZ}, vmax) where {FT, TX, TZ} = Inf
z_cell_advection_timescale(grid::RegularRectilinearGrid{FT, TX, TY, <:Flat}, wmax) where {FT, TX, TY} = Inf

"Returns the time-scale for advection on a regular grid across a single grid cell."
function cell_advection_timescale(u, v, w, grid)
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    return min(
        x_cell_advection_timescale(grid, umax),
        y_cell_advection_timescale(grid, vmax),
        z_cell_advection_timescale(grid, wmax),
       )
end


cell_advection_timescale(model) =
    cell_advection_timescale(model.velocities.u.data.parent,
                             model.velocities.v.data.parent,
                             model.velocities.w.data.parent,
                             model.grid)
