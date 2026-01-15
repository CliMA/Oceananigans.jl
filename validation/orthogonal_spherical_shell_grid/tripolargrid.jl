using Oceananigans
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
# using Oceananigans.Grids: RightFaceFolded
using GLMakie

"""
    plot_grid!(ax, grid)

Plots the grid as a wireframe on the given axis `ax`.
Includes markers at the centers.
Extra lines are drawn for the edges of the interior grid and, optionally, the halo.
Extra markers are also drawn for the centers located on the edges of the interior grid, and, optionally, the edges of the halo.
"""
function plot_grid!(ax, grid; plothalo = false)
    (; λᶜᶜᵃ, λᶠᶠᵃ) = grid
    (; φᶜᶜᵃ, φᶠᶠᵃ) = grid

    for (color, linestyle, linewidth, iscenter, x, y) in zip(
            ((:black, 0.25), (:black, 0.1)),
            (:solid, :solid),
            (1, 0.5),
            (false, true),
            (λᶠᶠᵃ, λᶜᶜᵃ),
            (φᶠᶠᵃ, φᶜᶜᵃ)
        )

        Hx, Hy = -1 .* x.offsets
        Nx, Ny = size(x) .- 2 .* (Hx, Hy)
        xleft = x[1, 1:Ny]; xhaloleft = x[1 - Hx, :]
        yleft = y[1, 1:Ny]; yhaloleft = y[1 - Hx, :]
        xright = x[Nx, 1:Ny]; xhaloright = x[Nx + Hx, :]
        yright = y[Nx, 1:Ny]; yhaloright = y[Nx + Hx, :]
        xbottom = x[1:Nx, 1]; xhalobottom = x[:, 1 - Hy]
        ybottom = y[1:Nx, 1]; yhalobottom = y[:, 1 - Hy]
        xtop = x[1:Nx, Ny]; xhalotop = x[:, Ny + Hy]
        ytop = y[1:Nx, Ny]; yhalotop = y[:, Ny + Hy]

        ix, iy = plothalo ? (Colon(), Colon()) : (1:Nx, 1:Ny)
        wireframe!(ax, x[ix, iy], y[ix, iy], 0 * x[ix, iy]; color, linestyle, linewidth = 2 * linewidth) # <- plot in Cartesian (x,y) coords

        # Color lines differently for first and last
        if iscenter
            sc = scatter!(ax, x[ix, iy][:], y[ix, iy][:]; color) # <- plot in Spherical (lon,lat) coords
            translate!(sc, 0, 0, 50)
            for (x, y, color, marker, markersize, fillit, plotit) in zip(
                    (xleft, xbottom, xright, xtop, xhaloleft, xhalobottom, xhaloright, xhalotop),
                    (yleft, ybottom, yright, ytop, yhaloleft, yhalobottom, yhaloright, yhalotop),
                    (:green, :blue, :red, :orange, :green, :blue, :red, :orange),
                    (:ltriangle, :dtriangle, :rtriangle, :utriangle, :ltriangle, :dtriangle, :rtriangle, :utriangle),
                    (10, 10, 10, 10, 10, 10, 10, 10),
                    (true, true, true, true, false, false, false, false),
                    (true, true, true, true, plothalo, plothalo, plothalo, plothalo),
                )
                plotit || continue
                sc = scatter!(ax, x, y; color = fillit ? color : :transparent, marker, markersize, strokecolor = color, strokewidth = 1)
                translate!(sc, 0, 0, 100)
            end
        else
            for (x, y, color, linewidth, plotit) in zip(
                    (xleft, xbottom, xright, xtop, xhaloleft, xhalobottom, xhaloright, xhalotop),
                    (yleft, ybottom, yright, ytop, yhaloleft, yhalobottom, yhaloright, yhalotop),
                    (:green, :blue, :red, :orange, :green, :blue, :red, :orange),
                    (3, 3, 3, 3, 1, 1, 1, 1),
                    (true, true, true, true, plothalo, plothalo, plothalo, plothalo),
                )
                plotit || continue
                ln = lines!(ax, x, y; color, linewidth)
                translate!(ln, 0, 0, 100)
            end
        end
    end
    return
end

fig = Figure(size = (1200, 1200))
# fig = Figure(size = (600, 600))
# Nx, Ny, Nz = 10, 10, 1
Nx, Ny, Nz = 30, 15, 1

axopt = (
    xticks = -720:30:720,
    yticks = -360:30:360,
)
labelopt = (rotation = π / 2, tellheight = false, fontsize = 24)

fold_topology = RightCenterFolded
grid = TripolarGrid(; size = (Nx, Ny, Nz), fold_topology)
ax = Axis(fig[1, 1]; axopt...)
plot_grid!(ax, grid; plothalo = true)
Label(fig[1, 0]; text = "$fold_topology (U-point pivot)", labelopt...)

fold_topology = RightFaceFolded
grid = TripolarGrid(; size = (Nx, Ny, Nz), fold_topology)
ax = Axis(fig[2, 1]; axopt...)
plot_grid!(ax, grid; plothalo = true)
Label(fig[2, 0]; text = "$fold_topology (F-point pivot)", labelopt...)

fig
