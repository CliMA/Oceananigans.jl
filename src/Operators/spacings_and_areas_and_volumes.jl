using Oceananigans.Grids: Center, Face

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
##### Spacing!!
#####
#####

@inline Δxᶠᵃᵃ(i, j, k, grid) =  nothing
@inline Δxᶜᵃᵃ(i, j, k, grid) =  nothing
@inline Δyᵃᶠᵃ(i, j, k, grid) =  nothing
@inline Δyᵃᶜᵃ(i, j, k, grid) =  nothing
@inline Δzᵃᵃᶠ(i, j, k, grid) =  nothing
@inline Δzᵃᵃᶜ(i, j, k, grid) =  nothing

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

        @eval begin
            @inline $x_spacing_3D(i, j, k, grid) = $x_spacing_2D(i, j, k, grid)
            @inline $y_spacing_3D(i, j, k, grid) = $y_spacing_2D(i, j, k, grid)
            @inline $z_spacing_3D(i, j, k, grid) = $z_spacing_1D(i, j, k, grid)
        end
    end
end

#####
##### Rectilinear Grids (Flat grids already have Δ = 1)
#####

@inline Δxᶠᵃᵃ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δxᶠᵃᵃ[i]
@inline Δxᶜᵃᵃ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δxᶜᵃᵃ[i]
@inline Δyᵃᶠᵃ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δyᵃᶠᵃ[j]
@inline Δyᵃᶜᵃ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δyᵃᶜᵃ[j]
@inline Δzᵃᵃᶠ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δzᵃᵃᶠ[k]
@inline Δzᵃᵃᶜ(i, j, k, grid::RectilinearGrid)     =  @inbounds grid.Δzᵃᵃᶜ[k]

@inline Δxᶠᵃᵃ(i, j, k, grid::XRegRectilinearGrid) =  @inbounds grid.Δxᶠᵃᵃ
@inline Δxᶜᵃᵃ(i, j, k, grid::XRegRectilinearGrid) =  @inbounds grid.Δxᶜᵃᵃ
@inline Δyᵃᶠᵃ(i, j, k, grid::YRegRectilinearGrid) =  @inbounds grid.Δyᵃᶠᵃ
@inline Δyᵃᶜᵃ(i, j, k, grid::YRegRectilinearGrid) =  @inbounds grid.Δyᵃᶜᵃ
@inline Δzᵃᵃᶠ(i, j, k, grid::ZRegRectilinearGrid) =  @inbounds grid.Δzᵃᵃᶠ
@inline Δzᵃᵃᶜ(i, j, k, grid::ZRegRectilinearGrid) =  @inbounds grid.Δzᵃᵃᶜ

#####
##### LatitudeLongitudeGrid
#####

# P stands for precomputed metrics, F stands for on the fly calculation of metrics
# the general case is when all the directions are stretched
# X, Y and Z stands for the direction which is regular

const LLGP  = LatitudeLongitudeGrid
const LLGPX = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number} # no i-index for Δλ
const LLGPY = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number} # no j-index for Δφ

const LLGF  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing}
const LLGFX = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Number}
const LLGFY = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Any, <:Number}

const LLGZ  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}

## On the fly metrics

@inline hack_cosd(φ) = cos(π * φ / 180)
@inline hack_sind(φ) = sin(π * φ / 180)

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δλᶠᵃᵃ[i]) * hack_cosd(grid.φᵃᶜᵃ[j]) 
@inline Δxᶠᶜᵃ(i, j, k, grid::LLGFX) = grid.radius * deg2rad(grid.Δλᶠᵃᵃ)    * hack_cosd(grid.φᵃᶜᵃ[j]) 
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δλᶜᵃᵃ[i]) * hack_cosd(grid.φᵃᶠᵃ[j]) 
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGFX) = grid.radius * deg2rad(grid.Δλᶜᵃᵃ)    * hack_cosd(grid.φᵃᶠᵃ[j]) 
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δλᶠᵃᵃ[i]) * hack_cosd(grid.φᵃᶠᵃ[j]) 
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGFX) = grid.radius * deg2rad(grid.Δλᶠᵃᵃ)    * hack_cosd(grid.φᵃᶠᵃ[j]) 
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δλᶜᵃᵃ[i]) * hack_cosd(grid.φᵃᶜᵃ[j]) 
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGFX) = grid.radius * deg2rad(grid.Δλᶜᵃᵃ)    * hack_cosd(grid.φᵃᶜᵃ[j]) 

@inline Δyᶜᶠᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
@inline Δyᶜᶠᵃ(i, j, k, grid::LLGFY) = grid.radius * deg2rad(grid.Δφᵃᶠᵃ)
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGFY) = grid.radius * deg2rad(grid.Δφᵃᶜᵃ)

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

@inline Δyᶜᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid) = Δyᶠᶜᵃ(i, j, k, grid)
@inline Δyᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid) = Δyᶜᶠᵃ(i, j, k, grid)

## Δz and Az metric

@inline Δzᵃᵃᶠ(i, j, k, grid::LLGZ) = grid.Δzᵃᵃᶠ
@inline Δzᵃᵃᶜ(i, j, k, grid::LLGZ) = grid.Δzᵃᵃᶜ

@inline Δzᵃᵃᶠ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.Δzᵃᵃᶠ[k]
@inline Δzᵃᵃᶜ(i, j, k, grid::LatitudeLongitudeGrid) = @inbounds grid.Δzᵃᵃᶜ[k]

#####
#####  ConformalCubedSphereFaceGrid
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

#####
#####
##### Areas!!
#####
#####

for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)
    
    x_spacing_3D = Symbol(:Δx, LX, LY, LZ)
    y_spacing_3D = Symbol(:Δy, LX, LY, LZ)
    z_spacing_3D = Symbol(:Δz, LX, LY, LZ)
    
    x_area_3D = Symbol(:Ax, LX, LY, LZ)
    y_area_3D = Symbol(:Ay, LX, LY, LZ)
    z_area_3D = Symbol(:Az, LX, LY, LZ)

    @eval begin
        @inline $x_area_3D(i, j, k, grid) = $y_spacing_3D(i, j, k, grid) * $z_spacing_3D(i, j, k, grid)
        @inline $y_area_3D(i, j, k, grid) = $x_spacing_3D(i, j, k, grid) * $z_spacing_3D(i, j, k, grid)
        @inline $z_area_3D(i, j, k, grid) = $x_spacing_3D(i, j, k, grid) * $y_spacing_3D(i, j, k, grid)
    end
end


####
#### Special Areas for LatitudeLongitudeGrid and ConformalCubedSphereFaceGrid
####

@inline latitude_variables(LY) = LY == :ᶠ ? (:φᵃᶜᵃ, 0, -1) : (:φᵃᶠᵃ, 1, 0) 

for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)
    
    z_area_3D = Symbol(:Az, LX, LY, LZ)
    z_area_2D = Symbol(:Az, LX, LY, :ᵃ)

    φ, j₁, j₂ = latitude_variables(LY)
    Δλ        = Symbol(:Δλ, LX, :ᵃᵃ)

    @eval begin
        @inline $z_area_3D(i, j, k, grid::Union{LLGP, ConformalCubedSphereFaceGrid})  = @inbounds grid.$z_area_2D[i, j]
        @inline $z_area_3D(i, j, k, grid::LLGPX)                                      = @inbounds grid.$z_area_2D[j]
        @inline $z_area_3D(i, j, k, grid::LLGF)   = @inbounds grid.radius^2 * deg2rad(grid.$Δλ[i]) * (hack_sind(grid.$φ[j + j₁]) - hack_sind(grid.$φ[j + j₂]))
        @inline $z_area_3D(i, j, k, grid::LLGFX)  = @inbounds grid.radius^2 * deg2rad(grid.$Δλ)    * (hack_sind(grid.$φ[j + j₁]) - hack_sind(grid.$φ[j + j₂]))
    end
end


#####
#####
##### Volumes!!
#####
#####

@inline Vᶜᶜᶜ(i, j, k, grid) = Azᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
@inline Vᶠᶜᶜ(i, j, k, grid) = Azᶠᶜᶜ(i, j, k, grid) * Δzᶠᶜᶜ(i, j, k, grid)
@inline Vᶜᶠᶜ(i, j, k, grid) = Azᶜᶠᶜ(i, j, k, grid) * Δzᶜᶠᶜ(i, j, k, grid)
@inline Vᶜᶜᶠ(i, j, k, grid) = Azᶜᶜᶠ(i, j, k, grid) * Δzᶜᶜᶠ(i, j, k, grid)
@inline Vᶠᶠᶜ(i, j, k, grid) = Azᶠᶠᶜ(i, j, k, grid) * Δzᶠᶠᶜ(i, j, k, grid)
@inline Vᶠᶜᶠ(i, j, k, grid) = Azᶠᶜᶠ(i, j, k, grid) * Δzᶠᶜᶠ(i, j, k, grid)
@inline Vᶜᶠᶠ(i, j, k, grid) = Azᶜᶠᶠ(i, j, k, grid) * Δzᶜᶠᶠ(i, j, k, grid)
@inline Vᶠᶠᶠ(i, j, k, grid) = Azᶠᶠᶠ(i, j, k, grid) * Δzᶠᶠᶠ(i, j, k, grid)

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
volume(i, j, k, grid, ::Face,   ::Face,   ::Center) = Vᶠᶠᶜ(i, j, k, grid)
volume(i, j, k, grid, ::Face,   ::Center, ::Face)   = Vᶠᶜᶠ(i, j, k, grid)
volume(i, j, k, grid, ::Center, ::Face,   ::Face)   = Vᶜᶠᶠ(i, j, k, grid)
volume(i, j, k, grid, ::Face,   ::Face,   ::Face)   = Vᶠᶠᶠ(i, j, k, grid)
