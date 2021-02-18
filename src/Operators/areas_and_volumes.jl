"""
Notes on generalization:

The operators in this file have been defined with the intent of implementing a vertically
stretched Cartesian grid. To generalize these operators to more general grids, numerous
new grid spacing operators will have to be defined which can be combined in a combinatorial
fashion to define new area and volume operators.

Generalizing to the somewhat simple case of non-uniform spacing in the x, y, and z involves
defining two grid spacings per dimensions, e.g. Δxᶜᵃᵃ for the distance between adjacent cell
faces and Δxᶠᵃᵃ for the distance between adjacent cell centers.

Generalizing to the more complex case of locally orthogonal grids such as the cubed sphere
may involve defining more grid spacing operators, potentially up to eight per dimension,
although not all may be used in practice.
"""

using Oceananigans.Grids: AbstractRectilinearGrid, RegularCartesianGrid, VerticallyStretchedCartesianGrid

#####
##### Rectilinear grid lengths
#####

@inline Δx(i, j, k, grid::AbstractRectilinearGrid) = grid.Δx
@inline Δy(i, j, k, grid::AbstractRectilinearGrid) = grid.Δy

@inline ΔzC(i, j, k, grid::RegularCartesianGrid) = grid.Δz
@inline ΔzC(i, j, k, grid::VerticallyStretchedCartesianGrid) = @inbounds grid.ΔzC[k]

@inline ΔzF(i, j, k, grid::RegularCartesianGrid) = grid.Δz
@inline ΔzF(i, j, k, grid::VerticallyStretchedCartesianGrid) = @inbounds grid.ΔzF[k]

#####
##### Rectilinear areas
#####

@inline Axᵃᵃᶜ(i, j, k, grid) = Δy(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Axᵃᵃᶠ(i, j, k, grid) = Δy(i, j, k, grid) * ΔzC(i, j, k, grid)

@inline Ayᵃᵃᶜ(i, j, k, grid) = Δx(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Ayᵃᵃᶠ(i, j, k, grid) = Δx(i, j, k, grid) * ΔzC(i, j, k, grid)

@inline Azᵃᵃᵃ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

#####
##### Rectilinear volumes
#####

@inline Vᵃᵃᶜ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Vᵃᵃᶠ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzC(i, j, k, grid)

#####
##### Horizontally-curvilinear grid lengths
#####

@inline Δxᶜᶜᵃ(i, j, k, grid::AbstractRectilinearGrid) = grid.Δx
@inline Δxᶜᶠᵃ(i, j, k, grid::AbstractRectilinearGrid) = grid.Δx
@inline Δxᶠᶠᵃ(i, j, k, grid::AbstractRectilinearGrid) = grid.Δx
@inline Δxᶠᶜᵃ(i, j, k, grid::AbstractRectilinearGrid) = grid.Δx

@inline Δyᶜᶜᵃ(i, j, k, grid::AbstractRectilinearGrid) = grid.Δy
@inline Δyᶠᶜᵃ(i, j, k, grid::AbstractRectilinearGrid) = grid.Δy
@inline Δyᶜᶠᵃ(i, j, k, grid::AbstractRectilinearGrid) = grid.Δy
@inline Δyᶠᶠᵃ(i, j, k, grid::AbstractRectilinearGrid) = grid.Δy

#####
##### Horizontally-curvilinear grid areas
#####

@inline Azᶠᶠᵃ(i, j, k, grid::AbstractRectilinearGrid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)
@inline Azᶜᶠᵃ(i, j, k, grid::AbstractRectilinearGrid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)
@inline Azᶠᶠᵃ(i, j, k, grid::AbstractRectilinearGrid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)
@inline Azᶠᶜᵃ(i, j, k, grid::AbstractRectilinearGrid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)
