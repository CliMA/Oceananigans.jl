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
lateral face down to the depth that the face leaves open. A cut column is bounded by its sloping bottom, while the other two are bounded by
the depth at the cell center:

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

titles = ("GridFittedBottom", "PartialCellBottom", "CutCellBottom")

for (n, immersed_grid) in enumerate((fitted_grid, partial_grid, cut_grid))
    ax = Axis(fig[1, n]; title=titles[n], aspect=1, xticks=xᶠ, yticks=zᶠ, limits=(0, 1, -1, 0))
    hidedecorations!(ax, grid=false)

    band!(ax, x, -1, slope.(x), color=(:tan, 0.6))

    ib = immersed_grid.immersed_boundary

    for i in 1:Nx
        if ib isa CutCellBottom
            west_height = ib.west_bottom_height[i, 1, 1]
            east_height = 2 * ib.bottom_height[i, 1, 1] - west_height
        else
            west_height = east_height = -static_column_depthᶜᶜᵃ(i, 1, immersed_grid)
        end

        column = Point2f[(xᶠ[i], west_height), (xᶠ[i+1], east_height), (xᶠ[i+1], 0), (xᶠ[i], 0)]
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

With `CutCellBottom`, the bottom of each column is a plane through the height of the column. When `bottom_height` is a function, it is
evaluated at the cell corners and the height of a column is the mean of its four corners. When it is an array or a field of cell-center
values, it gives the height of each column, and each corner is the mean of the wet columns around it. The plane is as steep as it can be
while each of its sides stays between the height of the column and the mean of the two corners on that side, so it is flat at a local
extremum of the bottom. Against a wall of the domain, only the side facing the interior bounds the slope. In ``x, z``, the plane of column
``i`` sits at ``r^W_i`` on its west side and at ``r^E_i = 2 r^b_i - r^W_i`` on its east side, where ``r^b_i`` is the height of the column.

The height of a cell is the mean of the openings its plane leaves at its lateral sides,

```math
\Delta z_{ccc}(i, k) = \frac{1}{2} \left[ o(r^W_i, k) + o(r^E_i, k) \right] \, ,
```

where ``o(r, k)`` is the height that a bottom at ``r`` leaves open in level ``k``, zero where the bottom lies above the level. A column is
therefore exactly as deep as its height, and every cut column in the figure above is a trapezoid. In three dimensions, the height of the cell
is the mean of the openings at its four lateral sides. A plane that spans more than one level cuts every cell it crosses, so a steep slope
cuts several cells of the same column.

The planes of two neighboring columns need not meet at their common face. A vertical wall then joins them, and the face opens down to the
higher of the two,

```math
\Delta z_{fcc}(i, k) = o \left( \max \left[ r^E_{i-1}, \, r^W_i \right], k \right) \, .
```

A face left with no opening in a level is peripheral and carries no flux, also between two active cells, as on a crest. Its spacing
``\Delta z_{fcc}`` keeps the height of the level, so that the velocity cell on it keeps a finite volume.

The opening the height of a column leaves in its level is rounded to the nearest of zero and ``\epsilon \Delta z``, where ``\epsilon`` is
the `minimum_fractional_cell_height`: a sliver thinner than ``\epsilon \Delta z / 2`` closes, and a thicker opening is at least
``\epsilon \Delta z`` tall. The slope of the plane is then reduced until neither of its sides leaves a thinner opening in any level, and the
plane falls flat when that does not settle. No side of a plane, and therefore no face, opens less than ``\epsilon \Delta z``. A cell averages
the openings of the sides of its plane, so it can be thinner, down to ``\epsilon \Delta z / 4``, or ``\epsilon \Delta z / 2`` in ``x, z``.
`CutCellBottom` is inspired by the shaved cells of [Adcroft et al. (1997)](@cite AdcroftHillMarshall1997).
