using Oceananigans.Grids: Center, Face
using Oceananigans.Grids: AbstractVerticalCoordinateUnderlyingGrid, ZStarUnderlyingGrid

@inline hack_cosd(φ) = cos(π * φ / 180)
@inline hack_sind(φ) = sin(π * φ / 180)

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
denoted by a triplet of superscripts. For example, an object `φ` whose cell is located at
(Center, Center, Face) is denoted `φᶜᶜᶠ`. `ᶜᶜᶠ` is Centered in `x`, `Centered` in `y`, and on
reference cell interfaces in `z` (this is where the vertical velocity is located, for example).

The operators in this file fall into three categories:

1. Operators needed for an algorithm valid on rectilinear grids with
   at most a stretched vertical dimension and regular horizontal dimensions.
2. Operators needed for an algorithm on a grid that is curvilinear in the horizontal
   at rectilinear (possibly stretched) in the vertical.
"""

#####
#####
##### Spacings!!
#####
#####

@inline Δxᶠᵃᵃ(i, j, k, grid) = nothing
@inline Δxᶜᵃᵃ(i, j, k, grid) = nothing
@inline Δyᵃᶠᵃ(i, j, k, grid) = nothing
@inline Δyᵃᶜᵃ(i, j, k, grid) = nothing

const ZRG = Union{LLGZ, RGZ, OSSGZ}

@inline Δzᵃᵃᶠ(i, j, k, grid) = @inbounds grid.Δzᵃᵃᶠ[k]
@inline Δzᵃᵃᶜ(i, j, k, grid) = @inbounds grid.Δzᵃᵃᶜ[k]

@inline Δzᵃᵃᶠ(i, j, k, grid::ZRG) = grid.Δzᵃᵃᶠ
@inline Δzᵃᵃᶜ(i, j, k, grid::ZRG) = grid.Δzᵃᵃᶜ

# Reference spacing, they differ only for a ZStar vertical coordinate grid
@inline Δrᵃᵃᶜ(i, j, k, grid) = Δzᵃᵃᶜ(i, j, k, grid)
@inline Δrᵃᵃᶠ(i, j, k, grid) = Δzᵃᵃᶠ(i, j, k, grid)

@inline getspacing(k, Δz::AbstractVector) = @inbounds Δz[k]
@inline getspacing(k, Δz::Number)         = @inbounds Δz

@inline Δrᵃᵃᶜ(i, j, k, grid::ZStarUnderlyingGrid) = getspacing(k, grid.Δzᵃᵃᶜ.reference)
@inline Δrᵃᵃᶠ(i, j, k, grid::ZStarUnderlyingGrid) = getspacing(k, grid.Δzᵃᵃᶠ.reference)

# Convenience Functions for all grids
for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ)

    x_spacing_1D = Symbol(:Δx, LX, :ᵃ, :ᵃ)
    x_spacing_2D = Symbol(:Δx, LX, LY, :ᵃ)

    y_spacing_1D = Symbol(:Δy, :ᵃ, LY, :ᵃ)
    y_spacing_2D = Symbol(:Δy, LX, LY, :ᵃ)

    @eval begin
        @inline $x_spacing_2D(i, j, k, grid) = $x_spacing_1D(i, j, k, grid)
        @inline $y_spacing_2D(i, j, k, grid) = $y_spacing_1D(i, j, k, grid)
    end

    for LZ in (:ᶜ, :ᶠ)
        x_spacing_3D = Symbol(:Δx, LX, LY, LZ)
        y_spacing_3D = Symbol(:Δy, LX, LY, LZ)

        z_spacing_1D = Symbol(:Δz, :ᵃ, :ᵃ, LZ)
        z_spacing_3D = Symbol(:Δz, LX, LY, LZ)

        r_spacing_1D = Symbol(:Δr, :ᵃ, :ᵃ, LZ)
        r_spacing_3D = Symbol(:Δr, LX, LY, LZ)

        @eval begin
            @inline $x_spacing_3D(i, j, k, grid) = $x_spacing_2D(i, j, k, grid)
            @inline $y_spacing_3D(i, j, k, grid) = $y_spacing_2D(i, j, k, grid)
            @inline $z_spacing_3D(i, j, k, grid) = $z_spacing_1D(i, j, k, grid)
            @inline $r_spacing_3D(i, j, k, grid) = $r_spacing_1D(i, j, k, grid)
        end
    end
end

##### 
##### 3D spacings for AbstractVerticalCoordinate grids
#####

const AVCG = AbstractVerticalCoordinateUnderlyingGrid
const c = Center()
const f = Face()

@inline Δzᶜᶜᶠ(i, j, k, grid::AVCG) = @inbounds Δrᶜᶜᶠ(i, j, k, grid) * vertical_scaling(i, j, k, grid, c, c, f)
@inline Δzᶜᶜᶜ(i, j, k, grid::AVCG) = @inbounds Δrᶜᶜᶜ(i, j, k, grid) * vertical_scaling(i, j, k, grid, c, c, c)
@inline Δzᶠᶜᶠ(i, j, k, grid::AVCG) = @inbounds Δrᶠᶜᶠ(i, j, k, grid) * vertical_scaling(i, j, k, grid, f, c, f)
@inline Δzᶠᶜᶜ(i, j, k, grid::AVCG) = @inbounds Δrᶠᶜᶜ(i, j, k, grid) * vertical_scaling(i, j, k, grid, f, c, c)
@inline Δzᶜᶠᶠ(i, j, k, grid::AVCG) = @inbounds Δrᶜᶠᶠ(i, j, k, grid) * vertical_scaling(i, j, k, grid, c, f, f)
@inline Δzᶜᶠᶜ(i, j, k, grid::AVCG) = @inbounds Δrᶜᶠᶜ(i, j, k, grid) * vertical_scaling(i, j, k, grid, c, f, c)
@inline Δzᶠᶠᶠ(i, j, k, grid::AVCG) = @inbounds Δrᶠᶠᶠ(i, j, k, grid) * vertical_scaling(i, j, k, grid, f, f, f)
@inline Δzᶠᶠᶜ(i, j, k, grid::AVCG) = @inbounds Δrᶠᶠᶜ(i, j, k, grid) * vertical_scaling(i, j, k, grid, f, f, c)

#####
##### Rectilinear Grids (Flat grids already have Δ = 1)
#####

@inline Δxᶠᵃᵃ(i, j, k, grid::RG) = @inbounds grid.Δxᶠᵃᵃ[i]
@inline Δxᶜᵃᵃ(i, j, k, grid::RG) = @inbounds grid.Δxᶜᵃᵃ[i]
@inline Δxᶜᵃᶜ(i, j, k, grid::RG) = @inbounds grid.Δxᶜᵃᵃ[i]
@inline Δxᶠᵃᶜ(i, j, k, grid::RG) = @inbounds grid.Δxᶠᵃᵃ[i]
@inline Δxᶜᵃᶠ(i, j, k, grid::RG) = @inbounds grid.Δxᶜᵃᵃ[i]

@inline Δyᵃᶠᵃ(i, j, k, grid::RG) = @inbounds grid.Δyᵃᶠᵃ[j]
@inline Δyᵃᶜᵃ(i, j, k, grid::RG) = @inbounds grid.Δyᵃᶜᵃ[j]
@inline Δyᶜᵃᶜ(i, j, k, grid::RG) = @inbounds grid.Δyᵃᶜᵃ[j]
@inline Δyᶠᵃᶜ(i, j, k, grid::RG) = @inbounds grid.Δyᵃᶜᵃ[j]
@inline Δyᶜᵃᶠ(i, j, k, grid::RG) = @inbounds grid.Δyᵃᶜᵃ[j]

@inline Δzᵃᵃᶠ(i, j, k, grid::RG) = @inbounds grid.Δzᵃᵃᶠ[k]
@inline Δzᵃᵃᶜ(i, j, k, grid::RG) = @inbounds grid.Δzᵃᵃᶜ[k]
@inline Δzᶜᵃᶜ(i, j, k, grid::RG) = @inbounds grid.Δzᵃᵃᶜ[k]
@inline Δzᶠᵃᶜ(i, j, k, grid::RG) = @inbounds grid.Δzᵃᵃᶜ[k]
@inline Δzᶜᵃᶠ(i, j, k, grid::RG) = @inbounds grid.Δzᵃᵃᶠ[k]

## XRegularRG

@inline Δxᶠᵃᵃ(i, j, k, grid::RGX) = grid.Δxᶠᵃᵃ
@inline Δxᶜᵃᵃ(i, j, k, grid::RGX) = grid.Δxᶜᵃᵃ

@inline Δxᶜᵃᶜ(i, j, k, grid::RGX) = grid.Δxᶜᵃᵃ
@inline Δxᶠᵃᶜ(i, j, k, grid::RGX) = grid.Δxᶠᵃᵃ
@inline Δxᶜᵃᶠ(i, j, k, grid::RGX) = grid.Δxᶜᵃᵃ

## YRegularRG

@inline Δyᵃᶠᵃ(i, j, k, grid::RGY) = grid.Δyᵃᶠᵃ
@inline Δyᵃᶜᵃ(i, j, k, grid::RGY) = grid.Δyᵃᶜᵃ

@inline Δyᶜᵃᶜ(i, j, k, grid::RGY) = grid.Δyᵃᶜᵃ
@inline Δyᶠᵃᶜ(i, j, k, grid::RGY) = grid.Δyᵃᶜᵃ
@inline Δyᶜᵃᶠ(i, j, k, grid::RGY) = grid.Δyᵃᶜᵃ

## ZRegularRG

@inline Δzᵃᵃᶠ(i, j, k, grid::RGZ) = grid.Δzᵃᵃᶠ
@inline Δzᵃᵃᶜ(i, j, k, grid::RGZ) = grid.Δzᵃᵃᶜ

@inline Δzᶜᵃᶜ(i, j, k, grid::RGZ) = grid.Δzᵃᵃᶜ
@inline Δzᶠᵃᶜ(i, j, k, grid::RGZ) = grid.Δzᵃᵃᶜ
@inline Δzᶜᵃᶠ(i, j, k, grid::RGZ) = grid.Δzᵃᵃᶠ

#####
##### LatitudeLongitudeGrid
#####

## Pre computed metrics

@inline Δxᶜᶠᵃ(i, j, k, grid::LLG) = @inbounds grid.Δxᶜᶠᵃ[i, j]
@inline Δxᶠᶜᵃ(i, j, k, grid::LLG) = @inbounds grid.Δxᶠᶜᵃ[i, j]
@inline Δxᶠᶠᵃ(i, j, k, grid::LLG) = @inbounds grid.Δxᶠᶠᵃ[i, j]
@inline Δxᶜᶜᵃ(i, j, k, grid::LLG) = @inbounds grid.Δxᶜᶜᵃ[i, j]

@inline Δyᶜᶠᵃ(i, j, k, grid::LLG) = @inbounds grid.Δyᶜᶠᵃ[j]
@inline Δyᶠᶜᵃ(i, j, k, grid::LLG) = @inbounds grid.Δyᶠᶜᵃ[j]
@inline Δyᶜᶜᵃ(i, j, k, grid::LLG) = Δyᶠᶜᵃ(i, j, k, grid)
@inline Δyᶠᶠᵃ(i, j, k, grid::LLG) = Δyᶜᶠᵃ(i, j, k, grid)

@inline Δzᵃᵃᶠ(i, j, k, grid::LLG) = @inbounds grid.Δzᵃᵃᶠ[k]
@inline Δzᵃᵃᶜ(i, j, k, grid::LLG) = @inbounds grid.Δzᵃᵃᶜ[k]

### XRegularLLG with pre-computed metrics

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶠᶜᵃ[j]
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶜᶠᵃ[j]
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶠᶠᵃ[j]
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶜᶜᵃ[j]

### YRegularLLG with pre-computed metrics

@inline Δyᶜᶠᵃ(i, j, k, grid::LLGY) = grid.Δyᶜᶠᵃ
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGY) = grid.Δyᶠᶜᵃ

### ZRegularLLG with pre-computed metrics

@inline Δzᵃᵃᶠ(i, j, k, grid::LLGZ) = grid.Δzᵃᵃᶠ
@inline Δzᵃᵃᶜ(i, j, k, grid::LLGZ) = grid.Δzᵃᵃᶜ

## On the fly metrics

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δλᶠᵃᵃ[i]) * hack_cosd(grid.φᵃᶜᵃ[j])
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δλᶜᵃᵃ[i]) * hack_cosd(grid.φᵃᶠᵃ[j])
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δλᶠᵃᵃ[i]) * hack_cosd(grid.φᵃᶠᵃ[j])
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δλᶜᵃᵃ[i]) * hack_cosd(grid.φᵃᶜᵃ[j])

@inline Δyᶜᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])

### XRegularLLG with on-the-fly metrics

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * deg2rad(grid.Δλᶠᵃᵃ)    * hack_cosd(grid.φᵃᶜᵃ[j])
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * deg2rad(grid.Δλᶜᵃᵃ)    * hack_cosd(grid.φᵃᶠᵃ[j])
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * deg2rad(grid.Δλᶠᵃᵃ)    * hack_cosd(grid.φᵃᶠᵃ[j])
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * deg2rad(grid.Δλᶜᵃᵃ)    * hack_cosd(grid.φᵃᶜᵃ[j])

### YRegularLLG with on-the-fly metrics

@inline Δyᶜᶠᵃ(i, j, k, grid::LLGFY) = grid.radius * deg2rad(grid.Δφᵃᶠᵃ)
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGFY) = grid.radius * deg2rad(grid.Δφᵃᶜᵃ)

#####
#####  OrthogonalSphericalShellGrid
#####

@inline Δxᶜᶜᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δxᶜᶜᵃ[i, j]
@inline Δxᶠᶜᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δxᶠᶜᵃ[i, j]
@inline Δxᶜᶠᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δxᶜᶠᵃ[i, j]
@inline Δxᶠᶠᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δxᶠᶠᵃ[i, j]

@inline Δyᶜᶜᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δyᶜᶜᵃ[i, j]
@inline Δyᶠᶜᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δyᶠᶜᵃ[i, j]
@inline Δyᶜᶠᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δyᶜᶠᵃ[i, j]
@inline Δyᶠᶠᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δyᶠᶠᵃ[i, j]

@inline Δzᵃᵃᶜ(i, j, k, grid::OSSG) = @inbounds grid.Δzᵃᵃᶜ[k]
@inline Δzᵃᵃᶠ(i, j, k, grid::OSSG) = @inbounds grid.Δzᵃᵃᶠ[k]

@inline Δzᵃᵃᶜ(i, j, k, grid::OSSGZ) = grid.Δzᵃᵃᶜ
@inline Δzᵃᵃᶠ(i, j, k, grid::OSSGZ) = grid.Δzᵃᵃᶠ

#####
#####
##### Areas!!
#####
#####

for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ)

    x_spacing_2D = Symbol(:Δx, LX, LY, :ᵃ)
    y_spacing_2D = Symbol(:Δy, LX, LY, :ᵃ)
    z_area_2D    = Symbol(:Az, LX, LY, :ᵃ)

    @eval $z_area_2D(i, j, k, grid) = $x_spacing_2D(i, j, k, grid) * $y_spacing_2D(i, j, k, grid)

    for LZ in (:ᶜ, :ᶠ)
        x_spacing_3D = Symbol(:Δx, LX, LY, LZ)
        y_spacing_3D = Symbol(:Δy, LX, LY, LZ)
        z_spacing_3D = Symbol(:Δz, LX, LY, LZ)

        x_area_3D = Symbol(:Ax, LX, LY, LZ)
        y_area_3D = Symbol(:Ay, LX, LY, LZ)
        z_area_3D = Symbol(:Az, LX, LY, LZ)

        @eval begin
            @inline $x_area_3D(i, j, k, grid) = $y_spacing_3D(i, j, k, grid) * $z_spacing_3D(i, j, k, grid)
            @inline $y_area_3D(i, j, k, grid) = $x_spacing_3D(i, j, k, grid) * $z_spacing_3D(i, j, k, grid)
            @inline $z_area_3D(i, j, k, grid) = $z_area_2D(i, j, k, grid)
        end
    end
end


####
#### Special 2D z Areas for LatitudeLongitudeGrid and OrthogonalSphericalShellGrid
####

@inline Azᶠᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶠᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ)    * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ)    * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ)    * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ)    * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))

for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ)

    z_area_2D = Symbol(:Az, LX, LY, :ᵃ)

    @eval begin
        @inline $z_area_2D(i, j, k, grid::OSSG) = @inbounds grid.$z_area_2D[i, j]
        @inline $z_area_2D(i, j, k, grid::LLG)  = @inbounds grid.$z_area_2D[i, j]
        @inline $z_area_2D(i, j, k, grid::LLGX) = @inbounds grid.$z_area_2D[j]
    end
end

#####
#####
##### Volumes!!
#####
#####

for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)

    volume = Symbol(:V, LX, LY, LZ)
    z_area = Symbol(:Az, LX, LY, LZ)
    z_spacing = Symbol(:Δz, LX, LY, LZ)

    @eval begin
        @inline $volume(i, j, k, grid) = $z_area(i, j, k, grid) * $z_spacing(i, j, k, grid)
    end
end

#####
##### Generic functions for specified locations
#####
##### For example, Δx(i, j, k, Face, Center, Face) is equivalent to = Δxᶠᶜᶠ(i, j, k, grid).
#####
##### We also use the function "volume" rather than `V`.
#####

location_code(LX, LY, LZ) = Symbol(interpolation_code(LX), interpolation_code(LY), interpolation_code(LZ))

for LX in (:Center, :Face, :Nothing)
    for LY in (:Center, :Face, :Nothing)
        for LZ in (:Center, :Face, :Nothing)
            LXe = @eval $LX
            LYe = @eval $LY
            LZe = @eval $LZ

            volume_function = Symbol(:V, location_code(LXe, LYe, LZe))
            @eval begin
                @inline volume(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $volume_function(i, j, k, grid)
            end

            for op in (:Δ, :A), dir in (:x, :y, :z)
                func   = Symbol(op, dir)
                metric = Symbol(op, dir, location_code(LXe, LYe, LZe))

                @eval begin
                    @inline $func(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $metric(i, j, k, grid)
                end
            end

            metric_function = Symbol(:Δr, location_code(LXe, LYe, LZe))
            @eval begin
                @inline Δr(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $metric_function(i, j, k, grid)
            end
        end
    end
end
