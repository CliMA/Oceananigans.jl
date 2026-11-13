# [Immersed boundaries](@id immersed_boundaries)

Irregular or "complex" domains, like an ocean basin with its bathymetry, are represented with [`ImmersedBoundaryGrid`](@ref).
An `ImmersedBoundaryGrid` embeds an immersed boundary into one of the underlying grids described in the [grids tutorial](@ref grids_tutorial).
The cells that lie inside the boundary are "immersed" and excluded from the computation. The immersed boundaries currently supported are:

1. [`GridFittedBottom`](@ref), which immerses every cell whose center lies below a bottom height, so that the bottom is a staircase of full cells.
2. [`PartialCellBottom`](@ref), which also reduces the height of the bottommost cell of each column to fit the bottom height.
3. [`CutCellBottom`](@ref), which also cuts the lateral faces of the bottommost cells where the bottom crosses them, so that the bottom slopes through each cell.
4. [`GridFittedBoundary`](@ref), which fits a three-dimensional mask to the grid.

## Three ways to fit a slope

To see how the three bottom immersed boundaries differ, we build a two-dimensional grid in ``x, z``, embed the same linear slope into it
with each of them, and draw the result. For every column we draw the water above the bottom that the grid sees, and a thick line over each
lateral face down to the depth that the face leaves open. A cut column is bounded by the depths of its two lateral faces, while the
other two are bounded by the depth at the cell center:

```@example immersed_boundaries
using Oceananigans
using Oceananigans.Grids: static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ
using CairoMakie

grid = RectilinearGrid(topology = (Bounded, Flat, Bounded),
                       size = (8, 6),
                       x = (0, 1),
                       z = (-1, 0))

slope(x) = -0.9 + 0.6x

fitted_grid  = ImmersedBoundaryGrid(grid, GridFittedBottom(slope))
partial_grid = ImmersedBoundaryGrid(grid, PartialCellBottom(slope))
cut_grid  = ImmersedBoundaryGrid(grid, CutCellBottom(slope))

Nx = size(grid, 1)
xᶠ = xnodes(grid, Face())
zᶠ = znodes(grid, Face())
x = range(0, 1, length=100)

fig = Figure(size=(1200, 450))

for (n, immersed_grid) in enumerate((fitted_grid, partial_grid, cut_grid))
    title = string(nameof(typeof(immersed_grid.immersed_boundary)))
    ax = Axis(fig[1, n]; title, aspect=1, xticks=xᶠ, yticks=zᶠ, limits=(0, 1, -1, 0))
    hidedecorations!(ax, grid=false)

    band!(ax, x, -1, slope.(x), color=(:tan, 0.6))

    for i in 1:Nx
        if immersed_grid.immersed_boundary isa CutCellBottom
            west_depth = static_column_depthᶠᶜᵃ(i, 1, immersed_grid)
            east_depth = static_column_depthᶠᶜᵃ(i+1, 1, immersed_grid)
        else
            west_depth = east_depth = static_column_depthᶜᶜᵃ(i, 1, immersed_grid)
        end

        column = Point2f[(xᶠ[i], -west_depth), (xᶠ[i+1], -east_depth), (xᶠ[i+1], 0), (xᶠ[i], 0)]
        poly!(ax, column, color=(:dodgerblue, 0.4))
    end

    for i in 2:Nx
        lines!(ax, [xᶠ[i], xᶠ[i]], [-static_column_depthᶠᶜᵃ(i, 1, immersed_grid), 0], color=:navy, linewidth=3)
    end

    lines!(ax, x, slope.(x), color=:black, linestyle=:dash)
end

fig
```

The thin lines mark the cells of the underlying grid and the dashed line is the slope.

### Grid-fitted bottoms

With `GridFittedBottom`, a cell is either fully immersed or fully open. The bottom sits on the interface of the underlying grid closest
to the bottom height, and the slope becomes a staircase with steps as tall as the vertical grid spacing.

### Partial cells

With `PartialCellBottom`, the bottommost cell of each column extends from the interface above it down to the bottom height. The height
of a partial cell is at least ``\epsilon \Delta z``, where ``\epsilon`` is the `minimum_fractional_cell_height` and ``\Delta z`` is the
height of the cell in the underlying grid. A lateral face, however, is only as tall as the shorter of the two cells it separates,

```math
\Delta z_{fcc}(i, k) = \min \left[ \Delta z_{ccc}(i-1, k), \, \Delta z_{ccc}(i, k) \right] \, ,
```

so a flow running along the slope squeezes through the steps of a staircase.

### Cut cells

With `CutCellBottom`, the bottom is a piecewise-bilinear surface defined by its heights at the cell corners. When `bottom_height` is a
function it is evaluated at the corners, and when it is an array of cell-center values it is interpolated to the corners. Every lateral
face of a bottom cell is cut by that surface where the face stands, so the faces of a cell have different heights. The height of the cell
is the mean of the openings its lateral faces leave in its level, which in ``x, z`` reads

```math
\Delta z_{ccc}(i, k) = \frac{1}{2} \left[ o(i, k) + o(i+1, k) \right] \, ,
```

where ``o(i, k)`` is the height of face ``i`` left open in level ``k``, zero where the face is closed. The faces therefore bound the volume
of the cell, and every cut column in the figure above is a trapezoid. In three dimensions, the height of the cell is the mean of the
openings of its four lateral faces. A closed face is peripheral and carries no flux; its spacing ``\Delta z_{fcc}`` keeps the height of the
level, so that the velocity cell on it keeps a finite volume.

Where the slope crosses from one level to the next, the face between the two columns is cut in the upper level and closed in the lower
one, so the cut cell in the lower level is a triangle. A slope steeper than one level per cell cuts the faces of a column in different
levels, so cut cells are stacked in a staircase.

The opening a face leaves in the lowest level it reaches is rounded to the nearest of zero and ``\epsilon \Delta z``: a sliver thinner than
``\epsilon \Delta z / 2`` closes the face in that level, and a thicker opening is at least ``\epsilon \Delta z`` tall. As for partial cells,
a cut cell is at least ``\epsilon \Delta z`` tall: when the mean opening of its faces is shorter, the faces cut in its level are deepened
until it is not, or until they leave the whole level open. `CutCellBottom` is inspired by the shaved cells of [Adcroft et al. (1997)](@cite AdcroftHillMarshall1997).
