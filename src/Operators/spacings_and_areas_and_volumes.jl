using Oceananigans.Grids: Center, Face
using KernelAbstractions
using Oceananigans.Architectures

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
The super script `ᵃ` denotes "any" location.

The operators in this file fall into three categories:

1. Operators needed for an algorithm valid on rectilinear grids with
   at most a stretched vertical dimension and regular horizontal dimensions.
2. Operators needed for an algorithm on a grid that is curvilinear in the horizontal
   at rectilinear (possibly stretched) in the vertical.

"""

#####
##### "Spacings" in Flat directions for rectilinear grids.
##### Here we dispatch all spacings to `one`. This abuse of notation
##### makes volumes and areas correct. E.g., with `z` Flat we have `v = dx * dy * 1`.
#####
##### Note: Vertical metrics are specific to each rectilinear grid type; therefore
##### we must dispatch on Flat vertical directions for each grid independently.
#####

using Oceananigans.Grids: Flat

#####
##### Grid lengths for horizontally-curvilinear, vertically-rectilinear algorithms
#####

@inline Δxᶜᶜᵃ(i, j, k, grid::ARG) = Δxᶜᵃᵃ(i, j, k, grid)
@inline Δxᶜᶠᵃ(i, j, k, grid::ARG) = Δxᶜᵃᵃ(i, j, k, grid)
@inline Δxᶠᶠᵃ(i, j, k, grid::ARG) = Δxᶠᵃᵃ(i, j, k, grid)
@inline Δxᶠᶜᵃ(i, j, k, grid::ARG) = Δxᶠᵃᵃ(i, j, k, grid)

@inline Δyᶜᶜᵃ(i, j, k, grid::ARG) = Δyᵃᶜᵃ(i, j, k, grid)
@inline Δyᶠᶜᵃ(i, j, k, grid::ARG) = Δyᵃᶜᵃ(i, j, k, grid)
@inline Δyᶜᶠᵃ(i, j, k, grid::ARG) = Δyᵃᶠᵃ(i, j, k, grid)
@inline Δyᶠᶠᵃ(i, j, k, grid::ARG) = Δyᵃᶠᵃ(i, j, k, grid)

#####
##### Areas for algorithms that generalize to horizontally-curvilinear, vertically-rectilinear grids
#####

@inline Azᶜᶜᵃ(i, j, k, grid::ARG) = Δxᶜᵃᵃ(i, j, k, grid) * Δyᵃᶜᵃ(i, j, k, grid)
@inline Azᶠᶠᵃ(i, j, k, grid::ARG) = Δxᶠᵃᵃ(i, j, k, grid) * Δyᵃᶠᵃ(i, j, k, grid)
@inline Azᶜᶠᵃ(i, j, k, grid::ARG) = Δxᶜᵃᵃ(i, j, k, grid) * Δyᵃᶠᵃ(i, j, k, grid)
@inline Azᶠᶜᵃ(i, j, k, grid::ARG) = Δxᶠᵃᵃ(i, j, k, grid) * Δyᵃᶜᵃ(i, j, k, grid)

#####
##### Areas for horizontally-regular grids
#####

@inline Axᵃᵃᶜ(i, j, k, grid::HRegRectilinearGrid) = Δyᵃᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
@inline Axᵃᵃᶠ(i, j, k, grid::HRegRectilinearGrid) = Δyᵃᶜᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid)

@inline Ayᵃᵃᶜ(i, j, k, grid::HRegRectilinearGrid) = Δxᶜᵃᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
@inline Ayᵃᵃᶠ(i, j, k, grid::HRegRectilinearGrid) = Δxᶜᵃᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid)

@inline Azᵃᵃᵃ(i, j, k, grid::HRegRectilinearGrid) = Δxᶜᵃᵃ(i, j, k, grid) * Δyᵃᶜᵃ(i, j, k, grid)

#####
##### Volumes for horizontally-regular algorithms
#####

@inline Vᵃᵃᶜ(i, j, k, grid::HRegRectilinearGrid) = Azᵃᵃᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
@inline Vᵃᵃᶠ(i, j, k, grid::HRegRectilinearGrid) = Azᵃᵃᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid)

#####
##### Areas for three-dimensionally curvilinear algorithms
#####

@inline Axᶜᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # c
@inline Axᶠᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶠᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # u
@inline Axᶠᶠᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶠᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # ζ
@inline Axᶜᶠᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶜᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # η
@inline Axᶠᶜᶠ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶠᶜᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid) # η
@inline Axᶜᶜᶠ(i, j, k, grid::Union{ARG, AHCG}) = Δyᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid) # η

@inline Ayᶜᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # c
@inline Ayᶜᶠᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶜᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # v
@inline Ayᶠᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶠᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # v
@inline Ayᶠᶠᶜ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶠᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid) # ζ
@inline Ayᶜᶠᶠ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶜᶠᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid) # ξ
@inline Ayᶜᶜᶠ(i, j, k, grid::Union{ARG, AHCG}) = Δxᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid) # ξ

#####
##### Volumes for three-dimensionally curvilinear algorithms
#####

@inline Vᶜᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Azᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
@inline Vᶠᶜᶜ(i, j, k, grid::Union{ARG, AHCG}) = Azᶠᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
@inline Vᶜᶠᶜ(i, j, k, grid::Union{ARG, AHCG}) = Azᶜᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
@inline Vᶜᶜᶠ(i, j, k, grid::Union{ARG, AHCG}) = Azᶜᶜᵃ(i, j, k, grid) * Δzᵃᵃᶠ(i, j, k, grid)

#####
##### Grid spacings and areas for RectilinearGrid
#####

@inline Δxᶜᵃᵃ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δxᶜᵃᵃ[i]
@inline Δxᶠᵃᵃ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δxᶠᵃᵃ[i]
@inline Δyᵃᶠᵃ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δyᵃᶠᵃ[j]
@inline Δyᵃᶜᵃ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δyᵃᶜᵃ[j]
@inline Δzᵃᵃᶠ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δzᵃᵃᶠ[k]
@inline Δzᵃᵃᶜ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δzᵃᵃᶜ[k]
@inline Δxᶜᵃᵃ(i, j, k, grid::XRegRectilinearGrid) =  @inbounds grid.Δxᶜᵃᵃ
@inline Δxᶠᵃᵃ(i, j, k, grid::XRegRectilinearGrid) =  @inbounds grid.Δxᶠᵃᵃ
@inline Δyᵃᶠᵃ(i, j, k, grid::YRegRectilinearGrid) =  @inbounds grid.Δyᵃᶠᵃ
@inline Δyᵃᶜᵃ(i, j, k, grid::YRegRectilinearGrid) =  @inbounds grid.Δyᵃᶜᵃ
@inline Δzᵃᵃᶠ(i, j, k, grid::ZRegRectilinearGrid) =  @inbounds grid.Δzᵃᵃᶠ
@inline Δzᵃᵃᶜ(i, j, k, grid::ZRegRectilinearGrid) =  @inbounds grid.Δzᵃᵃᶜ

const XFlatRG = RectilinearGrid{<:Any, <:Flat}
const YFlatRG = RectilinearGrid{<:Any, <:Any, <:Flat}
const ZFlatRG = RectilinearGrid{<:Any, <:Any, <:Any, <:Flat}

@inline Δxᶜᶜᵃ(i, j, k, grid::XFlatRG) = one(eltype(grid))
@inline Δxᶜᶠᵃ(i, j, k, grid::XFlatRG) = one(eltype(grid))
@inline Δxᶠᶠᵃ(i, j, k, grid::XFlatRG) = one(eltype(grid))
@inline Δxᶠᶜᵃ(i, j, k, grid::XFlatRG) = one(eltype(grid))
@inline Δyᶜᶜᵃ(i, j, k, grid::YFlatRG) = one(eltype(grid))
@inline Δyᶠᶜᵃ(i, j, k, grid::YFlatRG) = one(eltype(grid))
@inline Δyᶜᶠᵃ(i, j, k, grid::YFlatRG) = one(eltype(grid))
@inline Δyᶠᶠᵃ(i, j, k, grid::YFlatRG) = one(eltype(grid))
@inline Δzᵃᵃᶠ(i, j, k, grid::ZFlatRG) = one(eltype(grid))
@inline Δzᵃᵃᶜ(i, j, k, grid::ZFlatRG) = one(eltype(grid))

#####
##### Temporary place for grid spacings and areas for LatitudeLongitudeGrid
#####

""" 
the combination of types can be:

M <: Nothing mean no precomputed metrics. They have to be computed again.
FX<: Number  means that the grid is not stretched in the latitude direction
FY<: Number  means that the grid is not stretched in the longitude direction

"""

# P stands for precomputed metrics, F stands for on the fly calculation of metrics
# the general case is when all the directions are stretched
# X, Y and Z stands for the direction which is regular

const LLGP  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any}       # Actually == to LatitudeLongitudeGrid
const LLGPX = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number} # no i-index for Δλ
const LLGPY = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number} # no j-index for Δφ
const LLGF  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing}
const LLGFX = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Number}
const LLGFY = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Any, <:Number}
const LLGZ  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}

@inline hack_cosd(φ) = cos(π * φ / 180)
@inline hack_sind(φ) = sin(π * φ / 180)

## On the fly metrics

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ[i])
@inline Δxᶠᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ)
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ[i])
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ)
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ[i])
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ)
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ[i])
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ)

@inline Δyᶜᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
@inline Δyᶜᶠᵃ(i, j, k, grid::LLGFY) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ)
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGFY) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ)

@inline Azᶠᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶠᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ)    * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ)    * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ)    * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ)    * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))

## Pre computed metrics
## Δx metric

@inline Δxᶜᶠᵃ(i, j, k, grid::LLGP)  = @inbounds grid.Δxᶜᶠᵃ[i, j]
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGPX) = @inbounds grid.Δxᶜᶠᵃ[j]
@inline Δxᶠᶜᵃ(i, j, k, grid::LLGP)  = @inbounds grid.Δxᶠᶜᵃ[i, j]
@inline Δxᶠᶜᵃ(i, j, k, grid::LLGPX) = @inbounds grid.Δxᶠᶜᵃ[j]
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGP)  = @inbounds grid.Δxᶠᶠᵃ[i, j]
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGPX) = @inbounds grid.Δxᶠᶠᵃ[j]
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGP)  = @inbounds grid.Δxᶜᶜᵃ[i, j]
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGPX) = @inbounds grid.Δxᶜᶜᵃ[j]

## Δy metric

@inline Δyᶜᶠᵃ(i, j, k, grid::LLGP)  = @inbounds grid.Δyᶜᶠᵃ[j]
@inline Δyᶜᶠᵃ(i, j, k, grid::LLGPY) = @inbounds grid.Δyᶜᶠᵃ
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGP)  = @inbounds grid.Δyᶠᶜᵃ[j]
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGPY) = @inbounds grid.Δyᶠᶜᵃ
@inline Δyᶜᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid)  = Δyᶠᶜᵃ(i, j, k, grid)
@inline Δyᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid)  = Δyᶜᶠᵃ(i, j, k, grid)

## Δz and Az metric

@inline Δzᵃᵃᶜ(i, j, k, grid::LLGZ) = grid.Δzᵃᵃᶜ
@inline Δzᵃᵃᶠ(i, j, k, grid::LLGZ) = grid.Δzᵃᵃᶠ
@inline Δzᵃᵃᶜ(i, j, k, grid::LLGP) = @inbounds grid.Δzᵃᵃᶜ[k]
@inline Δzᵃᵃᶠ(i, j, k, grid::LLGP) = @inbounds grid.Δzᵃᵃᶠ[k]

@inline Azᶠᶜᵃ(i, j, k, grid::LLGP)  = @inbounds grid.Azᶠᶜᵃ[i, j]
@inline Azᶠᶜᵃ(i, j, k, grid::LLGPX) = @inbounds grid.Azᶠᶜᵃ[j]
@inline Azᶜᶠᵃ(i, j, k, grid::LLGP)  = @inbounds grid.Azᶜᶠᵃ[i, j]
@inline Azᶜᶠᵃ(i, j, k, grid::LLGPX) = @inbounds grid.Azᶜᶠᵃ[j]
@inline Azᶠᶠᵃ(i, j, k, grid::LLGP)  = @inbounds grid.Azᶠᶠᵃ[i, j]
@inline Azᶠᶠᵃ(i, j, k, grid::LLGPX) = @inbounds grid.Azᶠᶠᵃ[j]
@inline Azᶜᶜᵃ(i, j, k, grid::LLGP)  = @inbounds grid.Azᶜᶜᵃ[i, j]
@inline Azᶜᶜᵃ(i, j, k, grid::LLGPX) = @inbounds grid.Azᶜᶜᵃ[j]

#####
##### Temporary place for grid spacings and areas for ConformalCubedSphereFaceGrid
#####

@inline Δxᶜᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δxᶜᶜᵃ[i, j]
@inline Δxᶠᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δxᶠᶜᵃ[i, j]
@inline Δxᶜᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δxᶜᶠᵃ[i, j]
@inline Δxᶠᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δxᶠᶠᵃ[i, j]

@inline Δyᶜᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δyᶜᶜᵃ[i, j]
@inline Δyᶠᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δyᶠᶜᵃ[i, j]
@inline Δyᶜᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δyᶜᶠᵃ[i, j]
@inline Δyᶠᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Δyᶠᶠᵃ[i, j]

@inline Δzᵃᵃᶜ(i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.Δz
@inline Δzᵃᵃᶠ(i, j, k, grid::ConformalCubedSphereFaceGrid) = grid.Δz

@inline Azᶜᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Azᶜᶜᵃ[i, j]
@inline Azᶠᶜᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Azᶠᶜᵃ[i, j]
@inline Azᶜᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Azᶜᶠᵃ[i, j]
@inline Azᶠᶠᵃ(i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.Azᶠᶠᵃ[i, j]

#####
##### Generic functions for specified locations
#####
##### For example, Δx(i, j, k, Face, Center, LZ) is equivalent to = Δxᶠᶜᵃ(i, j, k, grid).
#####
##### We also use the function "volume" rather than `V`.
#####

location_code_xy(LX, LY) = Symbol(interpolation_code(LX), interpolation_code(LY), :ᵃ)
location_code(LX, LY, LZ) = Symbol(interpolation_code(LX), interpolation_code(LY), interpolation_code(LZ))

for LX in (:Center, :Face)
    for LY in (:Center, :Face)
        LXe = @eval $LX
        LYe = @eval $LY

        Ax_function = Symbol(:Ax, location_code(LXe, LYe, Center()))
        Ay_function = Symbol(:Ay, location_code(LXe, LYe, Center()))
        Az_function = Symbol(:Az, location_code_xy(LXe, LYe))

        @eval begin
            Az(i, j, k, grid, ::$LX, ::$LY, LZ) = $Az_function(i, j, k, grid)
            Ax(i, j, k, grid, ::$LX, ::$LY, ::Center) = $Ax_function(i, j, k, grid)
            Ay(i, j, k, grid, ::$LX, ::$LY, ::Center) = $Ay_function(i, j, k, grid)
        end
    end
end

Ax(i, j, k, grid, ::Face, ::Center, ::Face) = Axᶠᶜᶠ(i, j, k, grid)
Ay(i, j, k, grid, ::Center, ::Face, ::Face) = Ayᶜᶠᶠ(i, j, k, grid)

volume(i, j, k, grid, ::Center, ::Center, ::Center) = Vᶜᶜᶜ(i, j, k, grid)
volume(i, j, k, grid, ::Face,   ::Center, ::Center) = Vᶠᶜᶜ(i, j, k, grid)
volume(i, j, k, grid, ::Center, ::Face,   ::Center) = Vᶜᶠᶜ(i, j, k, grid)
volume(i, j, k, grid, ::Center, ::Center, ::Face)   = Vᶜᶜᶠ(i, j, k, grid)
