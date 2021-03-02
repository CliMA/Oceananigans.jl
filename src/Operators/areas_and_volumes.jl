"""
Notes:

This file defines grid lengths, areas, and volumes for staggered structured grids.

Each "reference cell" is associated with an index i, j, k.
The "location" of each reference cell is roughly the geometric centroid of the reference cell.

On the staggered grid, there are 7 cells additional to the "reference cell"
that are staggered with respect to the reference cell in x, y, and/or z.
The staggering is denoted by the locations "Center" and "Face":

    - "Center" is shared with the reference cell;
    - "Face" lies between reference cell centers, roughly at the interface between
      reference cells.

The three-dimensional location of an object is defined by a 3-tuple of locations, and
denoted by a triplet of superscripts. For example, an object `ϕ` whose cell is located at
(Center, Center, Face) is denoted `ϕᶜᶜᶠ`. `ᶜᶜᶠ` is Centered in `x`, `Centered` in `y`, and on
reference cell interfaces in `z` (this is where the vertical velocity is located, for example).

The super script `ᵃ` denotes "any" location.

The operators in this file fall into three categories:

1. Operators needed for an algorithm valid on rectilinear grids with
   at most a stretched vertical dimension and regular horizontal dimensions.

2. Operators needed for an algorithm on a grid that is curvilinear in the horizontal
   at rectilinear (possibly stretched) in the vertical.
"""

#####
##### Grid lengths for horiontally-regular algorithms
#####

@inline Δx(i, j, k, grid::ARG) = grid.Δx
@inline Δy(i, j, k, grid::ARG) = grid.Δy

@inline ΔzC(i, j, k, grid::RegularRectilinearGrid) = grid.Δz
@inline ΔzC(i, j, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.ΔzC[k]

@inline ΔzF(i, j, k, grid::RegularRectilinearGrid) = grid.Δz
@inline ΔzF(i, j, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.ΔzF[k]

@inline Δzᵃᵃᶠ(i, j, k, grid::RegularRectilinearGrid) = grid.Δz
@inline Δzᵃᵃᶠ(i, j, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.ΔzC[k]

@inline Δzᵃᵃᶜ(i, j, k, grid::RegularRectilinearGrid) = grid.Δz
@inline Δzᵃᵃᶜ(i, j, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.ΔzF[k]

#####
##### Areas for horiontally-regular algorithms
#####

@inline Axᵃᵃᶜ(i, j, k, grid) = Δy(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Axᵃᵃᶠ(i, j, k, grid) = Δy(i, j, k, grid) * ΔzC(i, j, k, grid)

@inline Ayᵃᵃᶜ(i, j, k, grid) = Δx(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Ayᵃᵃᶠ(i, j, k, grid) = Δx(i, j, k, grid) * ΔzC(i, j, k, grid)

@inline Azᵃᵃᵃ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

#####
##### Volumes for horiontally-regular algorithms
#####

@inline Vᵃᵃᶜ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzF(i, j, k, grid)
@inline Vᵃᵃᶠ(i, j, k, grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * ΔzC(i, j, k, grid)

#####
##### Grid lengths for horizontally-curvilinear, vertically-rectilinear algorithms
#####

@inline Δxᶜᶜᵃ(i, j, k, grid::ARG) = grid.Δx
@inline Δxᶜᶠᵃ(i, j, k, grid::ARG) = grid.Δx
@inline Δxᶠᶠᵃ(i, j, k, grid::ARG) = grid.Δx
@inline Δxᶠᶜᵃ(i, j, k, grid::ARG) = grid.Δx

@inline Δyᶜᶜᵃ(i, j, k, grid::ARG) = grid.Δy
@inline Δyᶠᶜᵃ(i, j, k, grid::ARG) = grid.Δy
@inline Δyᶜᶠᵃ(i, j, k, grid::ARG) = grid.Δy
@inline Δyᶠᶠᵃ(i, j, k, grid::ARG) = grid.Δy


#####
##### Areas for horizontally-curvilinear, vertically-rectilinear algorithms
#####

@inline Azᶜᶜᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid) * Δy(i, j, k, grid)
@inline Azᶠᶠᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid) * Δy(i, j, k, grid)
@inline Azᶜᶠᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid) * Δy(i, j, k, grid)
@inline Azᶠᶜᵃ(i, j, k, grid::ARG) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

#####
##### Areas for three-dimensionally curvilinear algorithms
#####

@inline Axᶠᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶠᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
@inline Ayᶜᶠᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶜᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)

#####
##### Volumes for three-dimensionally curvilinear algorithms
#####

@inline Vᶜᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Azᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)

#####
##### Temporary place for grid spacings and areas for RegularLatitudeLongitudeGrid
#####

@inline Δxᶜᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.radius * cosd(grid.ϕᵃᶠᵃ[j]) * deg2rad(grid.Δλ)
@inline Δxᶠᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.radius * cosd(grid.ϕᵃᶜᵃ[j]) * deg2rad(grid.Δλ)
@inline Δxᶜᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Δxᶠᶜᵃ(i, j, k, grid)
@inline Δxᶠᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Δxᶜᶠᵃ(i, j, k, grid)

@inline Δyᶜᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.radius * deg2rad(grid.Δϕ)
@inline Δyᶠᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Δyᶜᶠᵃ(i, j, k, grid)
@inline Δyᶜᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Δyᶜᶠᵃ(i, j, k, grid)
@inline Δyᶠᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Δyᶜᶠᵃ(i, j, k, grid)

@inline Δzᵃᵃᶜ(i, j, k, grid::RegularLatitudeLongitudeGrid) = grid.Δz
@inline Δzᵃᵃᶠ(i, j, k, grid::RegularLatitudeLongitudeGrid) = grid.Δz

@inline Axᶠᶜᶜ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds Δyᶠᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
@inline Ayᶜᶠᶜ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds Δxᶜᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)

@inline Azᶜᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλ) * (sind(grid.ϕᵃᶠᵃ[j+1]) - sind(grid.ϕᵃᶠᵃ[j]))
@inline Azᶠᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.radius^2 * deg2rad(grid.Δλ) * (sind(grid.ϕᵃᶜᵃ[j])   - sind(grid.ϕᵃᶜᵃ[j-1]))
@inline Azᶠᶜᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Azᶜᶜᵃ(i, j, k, grid)
@inline Azᶜᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = Azᶠᶠᵃ(i, j, k, grid)
