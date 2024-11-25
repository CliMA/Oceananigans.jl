using Oceananigans.Grids: Center, Face

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

spacing_x_loc(dir) = dir == :x ? (:ᶜ, :ᶠ) : (:ᶜ, :ᶠ, :ᵃ) 
spacing_y_loc(dir) = dir == :y ? (:ᶜ, :ᶠ) : (:ᶜ, :ᶠ, :ᵃ) 
spacing_z_loc(dir) = dir == :z ? (:ᶜ, :ᶠ) : (:ᶜ, :ᶠ, :ᵃ) 

spacing_1D(::Val{:x}, LX, LY, LZ) = Symbol(:Δx, LX, :ᵃ, :ᵃ)
spacing_2D(::Val{:x}, LX, LY, LZ) = Symbol(:Δx, LX, LY, :ᵃ)
spacing_1D(::Val{:y}, LX, LY, LZ) = Symbol(:Δy, :ᵃ, LY, :ᵃ)
spacing_2D(::Val{:y}, LX, LY, LZ) = Symbol(:Δy, LX, LY, :ᵃ)
spacing_1D(::Val{:z}, LX, LY, LZ) = Symbol(:Δz, :ᵃ, :ᵃ, LZ)
spacing_2D(::Val{:z}, LX, LY, LZ) = Symbol(:Δz, :ᵃ, LY, LZ)

# Convenience Functions for all grids
# This metaprogramming loop defines all the allowed combinations of Δx, Δy, and Δz
# Note `:ᵃ` is not allowed for the location associated with the spacing
for dir in (:x, :y, :z)
    for LX in spacing_x_loc(dir), LY in spacing_y_loc(dir), LZ in spacing_z_loc(dir)
        spacing1D = spacing_1D(Val(dir), LX, LY, LZ)
        spacing2D = spacing_2D(Val(dir), LX, LY, LZ)
        spacing3D = Symbol(:Δ, dir, LX, LY, LZ)
        if spacing2D != spacing1D
            @eval @inline $spacing2D(i, j, k, grid) = $spacing1D(i, j, k, grid)
        end
        if spacing3D != spacing2D
            @eval @inline $spacing3D(i, j, k, grid) = $spacing2D(i, j, k, grid)
        end
    end
end

#####
##### Vertical spacings (same for all grids)
#####

@inline Δzᵃᵃᶜ(i, j, k, grid) = @inbounds grid.Δzᵃᵃᶜ[k]
@inline Δzᵃᵃᶠ(i, j, k, grid) = @inbounds grid.Δzᵃᵃᶠ[k]

@inline Δzᶜᵃᶜ(i, j, k, grid) = @inbounds grid.Δzᵃᵃᶜ[k]
@inline Δzᶠᵃᶜ(i, j, k, grid) = @inbounds grid.Δzᵃᵃᶜ[k]
@inline Δzᶜᵃᶠ(i, j, k, grid) = @inbounds grid.Δzᵃᵃᶠ[k]
@inline Δzᶠᵃᶠ(i, j, k, grid) = @inbounds grid.Δzᵃᵃᶠ[k]

@inline Δzᵃᵃᶜ(i, j, k, grid::ZRG) = grid.Δzᵃᵃᶜ
@inline Δzᵃᵃᶠ(i, j, k, grid::ZRG) = grid.Δzᵃᵃᶠ

@inline Δzᶜᵃᶜ(i, j, k, grid::ZRG) = grid.Δzᵃᵃᶜ
@inline Δzᶠᵃᶜ(i, j, k, grid::ZRG) = grid.Δzᵃᵃᶜ
@inline Δzᶜᵃᶠ(i, j, k, grid::ZRG) = grid.Δzᵃᵃᶠ
@inline Δzᶠᵃᶠ(i, j, k, grid::ZRG) = grid.Δzᵃᵃᶠ

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

### XRegularLLG with pre-computed metrics

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶠᶜᵃ[j]
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶜᶠᵃ[j]
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶠᶠᵃ[j]
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶜᶜᵃ[j]

### YRegularLLG with pre-computed metrics

@inline Δyᶜᶠᵃ(i, j, k, grid::LLGY) = grid.Δyᶜᶠᵃ
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGY) = grid.Δyᶠᶜᵃ

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

#####
#####
##### Areas!!
#####
#####

for LX in (:ᶜ, :ᶠ, :ᵃ), LY in (:ᶜ, :ᶠ, :ᵃ)

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
        end
    end
end

# Special curvilinear spacings for curvilinear grids
@inline Δλ(i, j, k, grid::LLG,  ::Center, ℓy, ℓz) = @inbounds grid.Δλᶜᵃᵃ[i]
@inline Δλ(i, j, k, grid::LLG,  ::Face,   ℓy, ℓz) = @inbounds grid.Δλᶠᵃᵃ[i]
@inline Δλ(i, j, k, grid::LLGX, ::Center, ℓy, ℓz) = @inbounds grid.Δλᶜᵃᵃ
@inline Δλ(i, j, k, grid::LLGX, ::Face,   ℓy, ℓz) = @inbounds grid.Δλᶠᵃᵃ

@inline Δφ(i, j, k, grid::LLG,  ::Center, ℓy, ℓz) = @inbounds grid.Δφᵃᶜᵃ[j]
@inline Δφ(i, j, k, grid::LLG,  ::Face,   ℓy, ℓz) = @inbounds grid.Δφᵃᶠᵃ[j]
@inline Δφ(i, j, k, grid::LLG,  ::Center, ℓy, ℓz) = @inbounds grid.Δφᵃᶜᵃ
@inline Δφ(i, j, k, grid::LLG,  ::Face,   ℓy, ℓz) = @inbounds grid.Δφᵃᶠᵃ

@inline Δλ(i, j, k, grid::OSSG, ::Center, ℓy, ℓz) = δxᶜᵃᵃ(i, j, k, grid, λnode, Face(),   ℓy, ℓz)
@inline Δλ(i, j, k, grid::OSSG, ::Face,   ℓy, ℓz) = δxᶜᵃᵃ(i, j, k, grid, λnode, Center(), ℓy, ℓz)

@inline Δφ(i, j, k, grid::OSSG, ::Center, ℓy, ℓz) = δyᵃᶜᵃ(i, j, k, grid, λnode, ℓx, Face(),   ℓz)
@inline Δφ(i, j, k, grid::OSSG, ::Face,   ℓy, ℓz) = δyᵃᶜᵃ(i, j, k, grid, λnode, ℓx, Center(), ℓz)
