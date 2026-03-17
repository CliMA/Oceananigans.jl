using Oceananigans

using Oceananigans.Grids: RightFaceFolded, RightCenterFolded, halo_size
using Oceananigans.BoundaryConditions: Zipper, FPivot, UPivot, fill_halo_regions!
using Oceananigans.BoundaryConditions: UPivotZipperBoundaryCondition, FPivotZipperBoundaryCondition


using CairoMakie
# using GeometryBasics

fold_topologies = (RightCenterFolded, RightFaceFolded)
locations = (Center(), Face())

for fold_topology in fold_topologies

    grid = TripolarGrid(; size = (8, 10, 1), fold_topology = fold_topology, halo = (2, 2, 2))
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    pivotjᶜ, pivotjᶠ = (fold_topology == RightFaceFolded) ? (Ny - 1/2, Ny) : (Ny, Ny + 1/2)
    pivotiᶜ, pivotiᶠ = Nx ÷ 2 + 1/2, Nx ÷ 2 + 1

    fig = Figure(size = (1200, 800))
    legenddone = false

    for (ilocx, locx) in enumerate(locations), (ilocy, locy) in enumerate(locations)

        @info "Vizualizing halo fill for fold topology $fold_topology and location ($locx, $locy)"

        ifield = Field((locx, locy, Center()), grid)
        set!(ifield, [I.I[1] for I in CartesianIndices(size(ifield))])
        fill_halo_regions!(ifield)

        jfield = Field((locx, locy, Center()), grid)
        set!(jfield, [I.I[2] for I in CartesianIndices(size(jfield))])
        fill_halo_regions!(jfield)

        # And another tracer to track which halo points are actually filled
        # by using the sign change (it's a bit of a hack, yes)
        north = (fold_topology == RightFaceFolded) ? FPivotZipperBoundaryCondition(-1) : UPivotZipperBoundaryCondition(-1)
        boundary_conditions = FieldBoundaryConditions(grid, (locx, locy, Center()); north)
        asymfield = Field((locx, locy, Center()), grid; boundary_conditions)
        set!(asymfield, 1)
        fill_halo_regions!(asymfield, 1)

        Nxfield, Nyfield, _ = size(ifield)

        ax = Axis(fig[ilocx, ilocy];
            title = "($locx, $locy)",
            aspect = DataAspect(),
            xticks = (1 - Hx):(Nxfield + Hx),
            yticks = (1 - Hy):(Nyfield + Hy),
        )


        # Plot pivot points in red
        pivoti = (locx == Center()) ? pivotiᶜ : pivotiᶠ
        pivotj = (locy == Center()) ? pivotjᶜ : pivotjᶠ
        scpivots = scatter!(ax, pivoti .+ [-Nx ÷ 2, 0, Nx ÷ 2], pivotj .+ [0, 0, 0];
        color = :red, marker = :star5, markersize = 15, label = "Pivot points"
        )
        # default grid underlay
        offset = (fold_topology == RightFaceFolded) ? 1 : 1/2 # how much more grid north of pivots
        band!(ax, [pivotj - Hy - 2, pivotj + offset],
            (pivoti - Nx ÷ 2) * [1, 1], (pivoti + Nx ÷ 2) * [1, 1],
            direction = :y, color = (:black, 0.1),
        )

        # Plot arcs from halo points to their source
        for i in (1-Hx):(Nxfield+Hx), j in (1-Hy):(Nyfield+Hy)
            asymfield[i, j, 1] == -1 || continue # Here is the hack: skip non-changed sign points
            source = (ifield[i, j, 1], jfield[i, j, 1])
            destination = (i, j)
            origin = (source .+ destination) ./ 2
            radius = sqrt(sum((source .- destination) .^ 2)) / 2
            start_angle = atan(destination[2] - origin[2], destination[1] - origin[1])
            stop_angle = start_angle - π
            scsrc = scatter!(ax, source...; color = Cycled(j), label = "Interior source points")
            # scatter!(ax, destination...; marker = :rect, color = Cycled(j), markersize = 20)
            scdest = scatter!(ax, destination...;
                marker = :rect, color = :white, markersize = 15,
                strokecolor = :black, strokewidth = 1, label = "Folded points",
            )
            translate!(scdest, 0, 0, 1) # Hack to put the white marker on top of the colored one
            # dtriangle = Polygon([Point(0, 0), Point(1, 1), Point(-1, 1)])
            scutri = scatter!(ax, destination...; marker = :dtriangle, rotation = stop_angle, color = Cycled(j))
            translate!(scutri, 0, 0, 2) # Hack more
            ar = arc!(ax, origin, radius, start_angle, stop_angle, color = Cycled(j))
            translate!(ar, 0, 0, 2) # Hack more

            # add legend once
            if !legenddone
                axislegend(ax, [scpivots, scsrc, scdest], ["Pivot points / north poles", "Interior \"source\" points", "Points filled when folding"]; position = :rt)
                legenddone = true
            end
        end

    end

    Label(fig[0, :], "$fold_topology ($Nx×$Ny) TripolarGrid with ($Hx, $Hy) halo: fold visualisation", fontsize = 20)

    save("halo_fill_viz_$(fold_topology).png", fig)

end