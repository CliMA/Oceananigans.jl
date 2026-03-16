using Oceananigans
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using Oceananigans.Grids: RightFaceFolded, topology
using GLMakie

"""
    plot_grid!(ax, grid)

Plots the grid as a wireframe on the given axis `ax`.
Includes markers at the centers.
Extra lines are drawn for the edges of the interior grid.
Extra markers are also drawn for the centers located on the edges of the interior grid.
"""
function plot_grid!(ax, grid)

    λᶜᶜᵃ_interior = λnodes(grid, Center(), Center())
    λᶠᶠᵃ_interior = λnodes(grid, Face(), Face())
    φᶜᶜᵃ_interior = φnodes(grid, Center(), Center())
    φᶠᶠᵃ_interior = φnodes(grid, Face(), Face())

    (; Hx, Hy) = grid

    for (color, linestyle, linewidth, iscenter, x, y) in zip(
            ((:black, 0.25), (:black, 0.1)),
            (:solid, :solid),
            (1, 0.5),
            (false, true),
            (λᶠᶠᵃ_interior, λᶜᶜᵃ_interior),
            (φᶠᶠᵃ_interior, φᶜᶜᵃ_interior),
        )

        Nx, Ny = size(x)
        @assert (Nx, Ny) == size(y)

        xleft = x[1, 1:Ny]
        yleft = y[1, 1:Ny]
        xright = x[Nx, 1:Ny]
        yright = y[Nx, 1:Ny]
        xbottom = x[1:Nx, 1]
        ybottom = y[1:Nx, 1]
        xtop = x[1:Nx, Ny]
        ytop = y[1:Nx, Ny]

        wireframe!(ax, x, y, 0 * x; color, linestyle, linewidth = 2 * linewidth) # <- plot in Cartesian (x,y) coords

        # Color lines differently for first and last
        if iscenter
            sc = scatter!(ax, x[:], y[:]; color) # <- plot in Spherical (lon,lat) coords
            translate!(sc, 0, 0, 50)
            for (x, y, color, marker, markersize) in zip(
                    (xleft, xbottom, xright, xtop),
                    (yleft, ybottom, yright, ytop),
                    (:green, :blue, :red, :orange),
                    (:ltriangle, :dtriangle, :rtriangle, :utriangle),
                    (10, 10, 10, 10),
                )
                sc = scatter!(ax, x, y; color, marker, markersize, strokecolor = color, strokewidth = 1)
                translate!(sc, 0, 0, 100)
            end
        else
            for (x, y, color, linewidth) in zip(
                    (xleft, xbottom, xright, xtop),
                    (yleft, ybottom, yright, ytop),
                    (:green, :blue, :red, :orange),
                    (6, 5, 4, 3),
                )
                ln = lines!(ax, x, y; color, linewidth)
                translate!(ln, 0, 0, 100)
            end
        end
    end

    (; first_pole_longitude, north_poles_latitude) = grid.conformal_mapping
    sc = scatter!(ax, first_pole_longitude .+ [0, 180, 360], north_poles_latitude .+ [0, 0, 0]; color = :purple, marker = :star5, markersize = 20)
    translate!(sc, 0, 0, 300)

    return
end

"""Determine Location from 3 characters at the end?"""
function celllocation(char::Char)
    return char == 'ᶜ' ? Center :
        char == 'ᶠ' ? Face :
        char == 'ᵃ' ? Center :
        throw(ArgumentError("Unknown cell location character: $char"))
end
function celllocation(str::String)
    N = ncodeunits(str)
    iz = prevind(str, N)
    z = celllocation(str[iz])
    iy = prevind(str, iz)
    y = celllocation(str[iy])
    ix = prevind(str, iy)
    x = celllocation(str[ix])
    return (x, y, z)
end
celllocation(sym::Symbol) = celllocation(String(sym))

"""
    plot_metric(grid, metric_symbol; prefix = "")

Plots a heatmap of the gien metric and saves it to a PNG file.
Adds a polygon representing the interior points.
"""
function plot_metric(grid, metric_symbol; prefix = "")
    xdata = getproperty(grid, metric_symbol)
    (Hx, Hy) = .-xdata.offsets
    (Nx, Ny) = Base.size(xdata) .- 2 .* (Hx, Hy)
    # location-referenced pivot point indices
    c_pivot_i = v_pivot_i = grid.Nx ÷ 2 + 0.5
    u_pivot_i = grid.Nx ÷ 2 + 1
    c_pivot_j = u_pivot_j = (topology(grid, 2) == RightCenterFolded) ? grid.Ny : grid.Ny + 0.5
    v_pivot_j = c_pivot_j + 0.5
    loc = celllocation(metric_symbol)
    pivot_i, pivot_j = if loc == (Center, Center, Center)
        c_pivot_i, c_pivot_j
    elseif loc == (Face, Face, Center)
        u_pivot_i, v_pivot_j
    elseif loc == (Center, Face, Center)
        c_pivot_i, v_pivot_j
    elseif loc == (Face, Center, Center)
        u_pivot_i, c_pivot_j
    end
    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel = "i",
        ylabel = "j",
        aspect = DataAspect(),
        xticks = [1 - Hx, 1, Nx, Nx + Hx],
        yticks = [1 - Hy, 1, Ny, Ny + Hy],
    )
    extraopt = (; nan_color = :gray)
    hm = heatmap!(ax, (1 - Hx):(Nx + Hx), (1 - Hy):(Ny + Hy), xdata[:, :].parent; extraopt...)
    pl = poly!(ax, [0.5, Nx + 0.5, Nx + 0.5, 0.5, 0.5], [0.5, 0.5, Ny + 0.5, Ny + 0.5, 0.5];
        color = (:red, 0.0),
        strokewidth = 2,
        linestyle = :solid,
        strokecolor = :red
    )
    scatter!(ax, [pivot_i], [pivot_j]; color = :purple, marker = :star5, markersize = 15)
    ax.title = "$(prefix) $metric_symbol"
    Colorbar(fig[2, 1], hm; vertical = false, tellwidth = false)
    save("$(prefix)_$(metric_symbol).png", fig)
    return fig
end


fig = Figure(size = (1600, 800))

Nx, Ny, Nz = 20, 15, 1
halo = (2, 2, 2)
southernmost_latitude = -80
first_pole_longitude = 70
north_poles_latitude = 55

gridopt = (;
    size = (Nx, Ny, Nz),
    halo,
    southernmost_latitude,
    first_pole_longitude,
    north_poles_latitude
)

axopt = (
    xticks = -720:90:720,
    yticks = -360:45:360,
)
xlabelopt = (rotation = π / 2, tellheight = false, fontsize = 24)
ylabelopt = (tellwidth = false, fontsize = 24)

fold_topologies = (RightCenterFolded, RightFaceFolded)
Nxs = (Nx, Nx + 2)
metrics = (
    :λᶜᶜᵃ, :φᶜᶜᵃ, :Δxᶜᶜᵃ, :Δyᶜᶜᵃ, :Azᶜᶜᵃ,
    :λᶜᶠᵃ, :φᶜᶠᵃ, :Δxᶜᶠᵃ, :Δyᶜᶠᵃ, :Azᶜᶠᵃ,
    :λᶠᶜᵃ, :φᶠᶜᵃ, :Δxᶠᶜᵃ, :Δyᶠᶜᵃ, :Azᶠᶜᵃ,
    :λᶠᶠᵃ, :φᶠᶠᵃ, :Δxᶠᶠᵃ, :Δyᶠᶠᵃ, :Azᶠᶠᵃ,
)

for (irow, fold_topology) in enumerate(fold_topologies)
    for (icol, Nx′) in enumerate(Nxs)
        @info "Plotting grid with fold topology: $fold_topology and Nx = $Nx′"
        grid = TripolarGrid(; gridopt..., fold_topology, size = (Nx′, Ny, Nz))
        ax = Axis(fig[irow, icol]; axopt...)
        plot_grid!(ax, grid)
        Label(fig[0, icol]; text = "Nx = $(Nx′)", ylabelopt...)

        # Also plot all the metrics in separate figures
        for (imetric, metric) in enumerate(metrics)
            local fig = Figure()
            plot_metric(grid, metric; prefix = "$(fold_topology)_Nx$(Nx)")
        end

    end
    Label(fig[irow, 0]; text = "$fold_topology", xlabelopt...)
end

save("tripolar_grids.png", fig)
