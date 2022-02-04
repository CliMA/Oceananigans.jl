using Oceananigans.Grids: Center, Face

const RG   = RectilinearGrid
const RGX  = XRegRectilinearGrid
const RGY  = YRegRectilinearGrid
const RGZ  = ZRegRectilinearGrid

const CCSG = ConformalCubedSphereFaceGrid

const LLG  = LatitudeLongitudeGrid
const LLGX = XRegLatLonGrid
const LLGY = YRegLatLonGrid
const LLGZ = ZRegLatLonGrid

# On the fly calculations of metrics
const LLGF  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing}
const LLGFX = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Number}
const LLGFY = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Any, <:Number}

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

@inline Δxᶠᵃᵃ(i, j, k, grid) =  nothing
@inline Δxᶜᵃᵃ(i, j, k, grid) =  nothing
@inline Δyᵃᶠᵃ(i, j, k, grid) =  nothing
@inline Δyᵃᶜᵃ(i, j, k, grid) =  nothing

ZRG = Union{LLGZ, RGZ}

@inline Δzᵃᵃᶠ(i, j, k, grid) = @inbounds grid.Δzᵃᵃᶠ[k]
@inline Δzᵃᵃᶜ(i, j, k, grid) = @inbounds grid.Δzᵃᵃᶜ[k]

@inline Δzᵃᵃᶠ(i, j, k, grid::ZRG) = @inbounds grid.Δzᵃᵃᶠ
@inline Δzᵃᵃᶜ(i, j, k, grid::ZRG) = @inbounds grid.Δzᵃᵃᶜ

@inline Δzᵃᵃᶜ(i, j, k, grid::CCSG) = grid.Δz
@inline Δzᵃᵃᶠ(i, j, k, grid::CCSG) = grid.Δz

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

@inline Δxᶠᵃᵃ(i, j, k, grid::RG)  =  @inbounds grid.Δxᶠᵃᵃ[i]
@inline Δxᶜᵃᵃ(i, j, k, grid::RG)  =  @inbounds grid.Δxᶜᵃᵃ[i]
@inline Δyᵃᶠᵃ(i, j, k, grid::RG)  =  @inbounds grid.Δyᵃᶠᵃ[j]
@inline Δyᵃᶜᵃ(i, j, k, grid::RG)  =  @inbounds grid.Δyᵃᶜᵃ[j]

@inline Δxᶠᵃᵃ(i, j, k, grid::RGX) =  @inbounds grid.Δxᶠᵃᵃ
@inline Δxᶜᵃᵃ(i, j, k, grid::RGX) =  @inbounds grid.Δxᶜᵃᵃ
@inline Δyᵃᶠᵃ(i, j, k, grid::RGY) =  @inbounds grid.Δyᵃᶠᵃ
@inline Δyᵃᶜᵃ(i, j, k, grid::RGY) =  @inbounds grid.Δyᵃᶜᵃ

#####
##### LatitudeLongitudeGrid
#####

## Pre computed metrics

@inline Δxᶜᶠᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δxᶜᶠᵃ[i, j]
@inline Δxᶠᶜᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δxᶠᶜᵃ[i, j]
@inline Δxᶠᶠᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δxᶠᶠᵃ[i, j]
@inline Δxᶜᶜᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δxᶜᶜᵃ[i, j]
@inline Δxᶠᶜᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶠᶜᵃ[j]
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶜᶠᵃ[j]
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶠᶠᵃ[j]
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶜᶜᵃ[j]

@inline Δyᶜᶠᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δyᶜᶠᵃ[j]
@inline Δyᶠᶜᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δyᶠᶜᵃ[j]
@inline Δyᶜᶠᵃ(i, j, k, grid::LLGY) = @inbounds grid.Δyᶜᶠᵃ
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGY) = @inbounds grid.Δyᶠᶜᵃ
@inline Δyᶜᶜᵃ(i, j, k, grid::LLG)  = Δyᶠᶜᵃ(i, j, k, grid)
@inline Δyᶠᶠᵃ(i, j, k, grid::LLG)  = Δyᶜᶠᵃ(i, j, k, grid)

## On the fly metrics

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δλᶠᵃᵃ[i]) * hack_cosd(grid.φᵃᶜᵃ[j]) 
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δλᶜᵃᵃ[i]) * hack_cosd(grid.φᵃᶠᵃ[j]) 
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δλᶠᵃᵃ[i]) * hack_cosd(grid.φᵃᶠᵃ[j]) 
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δλᶜᵃᵃ[i]) * hack_cosd(grid.φᵃᶜᵃ[j]) 
@inline Δxᶠᶜᵃ(i, j, k, grid::LLGFX) = grid.radius * deg2rad(grid.Δλᶠᵃᵃ)    * hack_cosd(grid.φᵃᶜᵃ[j]) 
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGFX) = grid.radius * deg2rad(grid.Δλᶜᵃᵃ)    * hack_cosd(grid.φᵃᶠᵃ[j]) 
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGFX) = grid.radius * deg2rad(grid.Δλᶠᵃᵃ)    * hack_cosd(grid.φᵃᶠᵃ[j]) 
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGFX) = grid.radius * deg2rad(grid.Δλᶜᵃᵃ)    * hack_cosd(grid.φᵃᶜᵃ[j]) 

@inline Δyᶜᶠᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGF)  = grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])
@inline Δyᶜᶠᵃ(i, j, k, grid::LLGFY) = grid.radius * deg2rad(grid.Δφᵃᶠᵃ)
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGFY) = grid.radius * deg2rad(grid.Δφᵃᶜᵃ)

#####
#####  ConformalCubedSphereFaceGrid
#####

@inline Δxᶜᶜᵃ(i, j, k, grid::CCSG) = @inbounds grid.Δxᶜᶜᵃ[i, j]
@inline Δxᶠᶜᵃ(i, j, k, grid::CCSG) = @inbounds grid.Δxᶠᶜᵃ[i, j]
@inline Δxᶜᶠᵃ(i, j, k, grid::CCSG) = @inbounds grid.Δxᶜᶠᵃ[i, j]
@inline Δxᶠᶠᵃ(i, j, k, grid::CCSG) = @inbounds grid.Δxᶠᶠᵃ[i, j]

@inline Δyᶜᶜᵃ(i, j, k, grid::CCSG) = @inbounds grid.Δyᶜᶜᵃ[i, j]
@inline Δyᶠᶜᵃ(i, j, k, grid::CCSG) = @inbounds grid.Δyᶠᶜᵃ[i, j]
@inline Δyᶜᶠᵃ(i, j, k, grid::CCSG) = @inbounds grid.Δyᶜᶠᵃ[i, j]
@inline Δyᶠᶠᵃ(i, j, k, grid::CCSG) = @inbounds grid.Δyᶠᶠᵃ[i, j]

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
            @inline $x_area_3D(i, j, k, grid) = $x_spacing_3D(i, j, k, grid) * $z_spacing_3D(i, j, k, grid)
            @inline $y_area_3D(i, j, k, grid) = $y_spacing_3D(i, j, k, grid) * $z_spacing_3D(i, j, k, grid)
            @inline $z_area_3D(i, j, k, grid) = $z_area_2D(i, j, k, grid)
        end
    end
end


####
#### Special 2D Areas for LatitudeLongitudeGrid and ConformalCubedSphereFaceGrid
####

@inline Azᶠᶜᵃ(i, j, k, grid::LLGF)  = grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLGF)  = grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGF)  = grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGF)  = grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶠᶜᵃ(i, j, k, grid::LLGFX) = grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ)    * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLGFX) = grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ)    * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGFX) = grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ)    * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGFX) = grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ)    * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))

for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ)
    
    z_area_2D = Symbol(:Az, LX, LY, :ᵃ)

    @eval begin
        @inline $z_area_2D(i, j, k, grid::CCSG) = @inbounds grid.$z_area_2D[i, j]
        @inline $z_area_2D(i, j, k, grid::LLG)  = @inbounds grid.$z_area_2D[i, j]
        @inline $z_area_2D(i, j, k, grid::LLGX) = @inbounds grid.$z_area_2D[j]
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
##### For example, Δx(i, j, k, Face, Center, Face) is equivalent to = Δxᶠᶜᶠ(i, j, k, grid).
#####
##### We also use the function "volume" rather than `V`.
#####

location_code(LX, LY, LZ) = Symbol(interpolation_code(LX), interpolation_code(LY), interpolation_code(LZ))

for LX in (:Center, :Face)
    for LY in (:Center, :Face)
        for LZ in (:Center, :Face)
            LXe = @eval $LX
            LYe = @eval $LY
            LZe = @eval $LZ

            Ax_function = Symbol(:Ax, location_code(LXe, LYe, LZe))
            Ay_function = Symbol(:Ay, location_code(LXe, LYe, LZe))
            Az_function = Symbol(:Az, location_code(LXe, LYe, LZe))

            volume_function = Symbol(:V, location_code(LXe, LYe, LZe))

            @eval begin
                Az(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $Az_function(i, j, k, grid)
                Ax(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $Ax_function(i, j, k, grid)
                Ay(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $Ay_function(i, j, k, grid)

                volume(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $volume_function(i, j, k, grid)
            end
        end
    end
end
