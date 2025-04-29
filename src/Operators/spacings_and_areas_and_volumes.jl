using Oceananigans.Grids: Center, Face, AbstractGrid

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

# This metaprogramming loop defines all possible combinations of locations for spacings.
# The general 2D and 3D spacings are reconducted to their one - dimensional counterparts.
# Grids that do not have a specific one - dimensional spacing for a given location need to
# extend these functions (for example, LatitudeLongitudeGrid).

# Calling a non existing function (for example Δxᶜᵃᶜ on an OrthogonalSphericalShellGrid) will throw an error because
# the associated one - dimensional function is not defined.
for L1 in (:ᶜ, :ᶠ), L2 in (:ᶜ, :ᶠ)
    Δxˡᵃᵃ = Symbol(:Δx, L1, :ᵃ, :ᵃ)
    Δyᵃˡᵃ = Symbol(:Δy, :ᵃ, L1, :ᵃ)
    Δzᵃᵃˡ = Symbol(:Δz, :ᵃ, :ᵃ, L1)
    Δλˡᵃᵃ = Symbol(:Δλ, L1, :ᵃ, :ᵃ)
    Δφᵃˡᵃ = Symbol(:Δφ, :ᵃ, L1, :ᵃ)
    Δrᵃᵃˡ = Symbol(:Δr, :ᵃ, :ᵃ, L1)

    Δxˡˡᵃ = Symbol(:Δx, L1, L2, :ᵃ)
    Δyˡˡᵃ = Symbol(:Δy, L2, L1, :ᵃ)
    Δzˡᵃˡ = Symbol(:Δz, L2, :ᵃ, L1)
    Δλˡˡᵃ = Symbol(:Δλ, L1, L2, :ᵃ)
    Δφˡˡᵃ = Symbol(:Δφ, L2, L1, :ᵃ)
    Δrˡᵃˡ = Symbol(:Δr, L2, :ᵃ, L1)

    Δxˡᵃˡ = Symbol(:Δx, L1, :ᵃ, L2)
    Δyᵃˡˡ = Symbol(:Δy, :ᵃ, L1, L2)
    Δzᵃˡˡ = Symbol(:Δz, :ᵃ, L2, L1)
    Δλˡᵃˡ = Symbol(:Δλ, L1, :ᵃ, L2)
    Δφᵃˡˡ = Symbol(:Δφ, :ᵃ, L1, L2)
    Δrᵃˡˡ = Symbol(:Δr, :ᵃ, L2, L1)

    @eval @inline $Δxˡˡᵃ(i, j, k, grid) = $Δxˡᵃᵃ(i, j, k, grid)
    @eval @inline $Δxˡᵃˡ(i, j, k, grid) = $Δxˡᵃᵃ(i, j, k, grid)

    @eval @inline $Δyˡˡᵃ(i, j, k, grid) = $Δyᵃˡᵃ(i, j, k, grid)
    @eval @inline $Δyᵃˡˡ(i, j, k, grid) = $Δyᵃˡᵃ(i, j, k, grid)

    @eval @inline $Δzˡᵃˡ(i, j, k, grid) = $Δzᵃᵃˡ(i, j, k, grid)
    @eval @inline $Δzᵃˡˡ(i, j, k, grid) = $Δzᵃᵃˡ(i, j, k, grid)

    @eval @inline $Δλˡˡᵃ(i, j, k, grid) = $Δλˡᵃᵃ(i, j, k, grid)
    @eval @inline $Δλˡᵃˡ(i, j, k, grid) = $Δλˡᵃᵃ(i, j, k, grid)

    @eval @inline $Δφˡˡᵃ(i, j, k, grid) = $Δφᵃˡᵃ(i, j, k, grid)
    @eval @inline $Δφᵃˡˡ(i, j, k, grid) = $Δφᵃˡᵃ(i, j, k, grid)

    @eval @inline $Δrˡᵃˡ(i, j, k, grid) = $Δrᵃᵃˡ(i, j, k, grid)
    @eval @inline $Δrᵃˡˡ(i, j, k, grid) = $Δrᵃᵃˡ(i, j, k, grid)

    for L3 in (:ᶜ, :ᶠ)
        Δxˡˡˡ = Symbol(:Δx, L1, L2, L3)
        Δyˡˡˡ = Symbol(:Δy, L2, L1, L3)
        Δzˡˡˡ = Symbol(:Δz, L2, L3, L1)
        Δλˡˡˡ = Symbol(:Δλ, L1, L2, L3)
        Δφˡˡˡ = Symbol(:Δφ, L2, L1, L3)
        Δrˡˡˡ = Symbol(:Δr, L2, L3, L1)

        @eval @inline $Δxˡˡˡ(i, j, k, grid) = $Δxˡˡᵃ(i, j, k, grid)
        @eval @inline $Δyˡˡˡ(i, j, k, grid) = $Δyˡˡᵃ(i, j, k, grid)
        @eval @inline $Δzˡˡˡ(i, j, k, grid) = $Δzˡᵃˡ(i, j, k, grid)
        @eval @inline $Δλˡˡˡ(i, j, k, grid) = $Δλˡˡᵃ(i, j, k, grid)
        @eval @inline $Δφˡˡˡ(i, j, k, grid) = $Δφˡˡᵃ(i, j, k, grid)
        @eval @inline $Δrˡˡˡ(i, j, k, grid) = $Δrˡᵃˡ(i, j, k, grid)
    end
end

#####
##### One - dimensional Vertical spacing (same for all grids)
#####

@inline getspacing(k, Δz::Number) = Δz
@inline getspacing(k, Δz::AbstractVector) = @inbounds Δz[k]

@inline Δrᵃᵃᶜ(i, j, k, grid) = getspacing(k, grid.z.Δᵃᵃᶜ)
@inline Δrᵃᵃᶠ(i, j, k, grid) = getspacing(k, grid.z.Δᵃᵃᶠ)

@inline Δzᵃᵃᶜ(i, j, k, grid) = getspacing(k, grid.z.Δᵃᵃᶜ)
@inline Δzᵃᵃᶠ(i, j, k, grid) = getspacing(k, grid.z.Δᵃᵃᶠ)

#####
#####
##### One - Dimensional Horizontal Spacings
#####
#####

#####
##### Rectilinear Grids (Flat grids already have Δ = 1)
#####

@inline Δxᶠᵃᵃ(i, j, k, grid::RG) = @inbounds grid.Δxᶠᵃᵃ[i]
@inline Δxᶜᵃᵃ(i, j, k, grid::RG) = @inbounds grid.Δxᶜᵃᵃ[i]

@inline Δyᵃᶠᵃ(i, j, k, grid::RG) = @inbounds grid.Δyᵃᶠᵃ[j]
@inline Δyᵃᶜᵃ(i, j, k, grid::RG) = @inbounds grid.Δyᵃᶜᵃ[j]

### XRegularRG

@inline Δxᶠᵃᵃ(i, j, k, grid::RGX) = grid.Δxᶠᵃᵃ
@inline Δxᶜᵃᵃ(i, j, k, grid::RGX) = grid.Δxᶜᵃᵃ


### YRegularRG

@inline Δyᵃᶠᵃ(i, j, k, grid::RGY) = grid.Δyᵃᶠᵃ
@inline Δyᵃᶜᵃ(i, j, k, grid::RGY) = grid.Δyᵃᶜᵃ

#####
##### LatitudeLongitude Grids (define both precomputed and non-precomputed metrics)
#####

### Curvilinear spacings

@inline Δλᶜᵃᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δλᶜᵃᵃ[i]
@inline Δλᶠᵃᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δλᶠᵃᵃ[i]
@inline Δλᶜᵃᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δλᶜᵃᵃ
@inline Δλᶠᵃᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δλᶠᵃᵃ

@inline Δφᵃᶜᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δφᵃᶜᵃ[j]
@inline Δφᵃᶠᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δφᵃᶠᵃ[j]
@inline Δφᵃᶜᵃ(i, j, k, grid::LLGY) = @inbounds grid.Δφᵃᶜᵃ
@inline Δφᵃᶠᵃ(i, j, k, grid::LLGY) = @inbounds grid.Δφᵃᶠᵃ

### Linear spacings

### Precomputed metrics

@inline Δyᵃᶜᵃ(i, j, k, grid::LLGY) = grid.Δyᶠᶜᵃ
@inline Δyᵃᶠᵃ(i, j, k, grid::LLGY) = grid.Δyᶜᶠᵃ
@inline Δyᵃᶜᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δyᶠᶜᵃ[j]
@inline Δyᵃᶠᵃ(i, j, k, grid::LLG)  = @inbounds grid.Δyᶜᶠᵃ[j]

### On-the-fly metrics

@inline Δyᵃᶠᵃ(i, j, k, grid::LLGFY) = grid.radius * deg2rad(grid.Δφᵃᶠᵃ)
@inline Δyᵃᶜᵃ(i, j, k, grid::LLGFY) = grid.radius * deg2rad(grid.Δφᵃᶜᵃ)
@inline Δyᵃᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
@inline Δyᵃᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])

#####
#####
##### Two-dimensional horizontal spacings
#####
#####

#####
##### LatitudeLongitudeGrid (only the Δx are required, Δy, Δλ, and Δφ are 1D)
#####

### Pre computed metrics

@inline Δxᶜᶠᵃ(i, j, k, grid::LLG) = @inbounds grid.Δxᶜᶠᵃ[i, j]
@inline Δxᶠᶜᵃ(i, j, k, grid::LLG) = @inbounds grid.Δxᶠᶜᵃ[i, j]
@inline Δxᶠᶠᵃ(i, j, k, grid::LLG) = @inbounds grid.Δxᶠᶠᵃ[i, j]
@inline Δxᶜᶜᵃ(i, j, k, grid::LLG) = @inbounds grid.Δxᶜᶜᵃ[i, j]

### XRegularLLG with pre computed metrics

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶠᶜᵃ[j]
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶜᶠᵃ[j]
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶠᶠᵃ[j]
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGX) = @inbounds grid.Δxᶜᶜᵃ[j]

### On-the-fly metrics

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGF) = @inbounds grid.radius * deg2rad(grid.Δλᶠᵃᵃ[i]) * hack_cosd(grid.φᵃᶜᵃ[j])
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGF) = @inbounds grid.radius * deg2rad(grid.Δλᶜᵃᵃ[i]) * hack_cosd(grid.φᵃᶠᵃ[j])
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGF) = @inbounds grid.radius * deg2rad(grid.Δλᶠᵃᵃ[i]) * hack_cosd(grid.φᵃᶠᵃ[j])
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGF) = @inbounds grid.radius * deg2rad(grid.Δλᶜᵃᵃ[i]) * hack_cosd(grid.φᵃᶜᵃ[j])

### XRegularLLG with on-the-fly metrics

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * deg2rad(grid.Δλᶠᵃᵃ) * hack_cosd(grid.φᵃᶜᵃ[j])
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * deg2rad(grid.Δλᶜᵃᵃ) * hack_cosd(grid.φᵃᶠᵃ[j])
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * deg2rad(grid.Δλᶠᵃᵃ) * hack_cosd(grid.φᵃᶠᵃ[j])
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * deg2rad(grid.Δλᶜᵃᵃ) * hack_cosd(grid.φᵃᶜᵃ[j])

#####
#####  OrthogonalSphericalShellGrid (does not have one-dimensional spacings)
#####

### Curvilinear spacings

@inline Δλᶜᶜᵃ(i, j, k, grid::OSSG) = δxᶜᵃᵃ(i, j, k, grid, λnode, Face(),   Center(), nothing)
@inline Δλᶠᶜᵃ(i, j, k, grid::OSSG) = δxᶠᵃᵃ(i, j, k, grid, λnode, Center(), Center(), nothing)
@inline Δλᶜᶠᵃ(i, j, k, grid::OSSG) = δxᶜᵃᵃ(i, j, k, grid, λnode, Face(),   Face(),   nothing)
@inline Δλᶠᶠᵃ(i, j, k, grid::OSSG) = δxᶠᵃᵃ(i, j, k, grid, λnode, Center(), Face(),   nothing)

@inline Δφᶜᶜᵃ(i, j, k, grid::OSSG) = δyᵃᶜᵃ(i, j, k, grid, λnode, Center(), Face(),   nothing)
@inline Δφᶠᶜᵃ(i, j, k, grid::OSSG) = δyᵃᶜᵃ(i, j, k, grid, λnode, Face(),   Face(),   nothing)
@inline Δφᶜᶠᵃ(i, j, k, grid::OSSG) = δyᵃᶠᵃ(i, j, k, grid, λnode, Center(), Center(), nothing)
@inline Δφᶠᶠᵃ(i, j, k, grid::OSSG) = δyᵃᶠᵃ(i, j, k, grid, λnode, Face(),   Center(), nothing)

### Linear spacings

@inline Δxᶜᶜᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δxᶜᶜᵃ[i, j]

@inline Δxᶜᶜᵃ(i::AbstractArray, j::AbstractArray, k::AbstractArray, grid::OSSG) = Base.stack(collect(Δxᶜᶜᵃ(i, j, 1, grid) for _ in k))

@inline Δxᶠᶜᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δxᶠᶜᵃ[i, j]
@inline Δxᶠᶜᵃ(i::AbstractArray, j::AbstractArray, k::AbstractArray, grid::OSSG) = Base.stack(collect(Δxᶠᶜᵃ(i, j, 1, grid) for _ in k))

@inline Δxᶜᶠᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δxᶜᶠᵃ[i, j]
@inline Δxᶜᶠᵃ(i::AbstractArray, j::AbstractArray, k::AbstractArray, grid::OSSG) = Base.stack(collect(Δxᶜᶠᵃ(i, j, 1, grid) for _ in k))

@inline Δxᶠᶠᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δxᶠᶠᵃ[i, j]
@inline Δxᶠᶠᵃ(i::AbstractArray, j::AbstractArray, k::AbstractArray, grid::OSSG) = Base.stack(collect(Δxᶠᶠᵃ(i, j, 1, grid) for _ in k))

@inline Δyᶜᶜᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δyᶜᶜᵃ[i, j]
@inline Δyᶜᶜᵃ(i::AbstractArray, j::AbstractArray, k::AbstractArray, grid::OSSG) = Base.stack(collect(Δyᶜᶜᵃ(i, j, 1, grid) for _ in k))

@inline Δyᶠᶜᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δyᶠᶜᵃ[i, j]
@inline Δyᶠᶜᵃ(i::AbstractArray, j::AbstractArray, k::AbstractArray, grid::OSSG) = Base.stack(collect(Δyᶠᶜᵃ(i, j, 1, grid) for _ in k))

@inline Δyᶜᶠᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δyᶜᶠᵃ[i, j]
@inline Δyᶜᶠᵃ(i::AbstractArray, j::AbstractArray, k::AbstractArray, grid::OSSG) = Base.stack(collect(Δyᶜᶠᵃ(i, j, 1, grid) for _ in k))

@inline Δyᶠᶠᵃ(i, j, k, grid::OSSG) = @inbounds grid.Δyᶠᶠᵃ[i, j]
@inline Δyᶠᶠᵃ(i::AbstractArray, j::AbstractArray, k::AbstractArray, grid::OSSG) = Base.stack(collect(Δyᶠᶠᵃ(i, j, 1, grid) for _ in k))

#####
#####
##### Areas!!
#####
#####

# We do the same thing as for the spacings: define general areas and then specialize for each grid.
# Areas need to be at least 2D so we use the respective 2D spacings to define them.
for L1 in (:ᶜ, :ᶠ), L2 in (:ᶜ, :ᶠ)

    Δxˡˡᵃ = Symbol(:Δx, L1, L2, :ᵃ)
    Δxˡᵃˡ = Symbol(:Δx, L1, :ᵃ, L2)
    Δyˡˡᵃ = Symbol(:Δy, L1, L2, :ᵃ)
    Δyᵃˡˡ = Symbol(:Δy, :ᵃ, L1, L2)
    Δzˡᵃˡ = Symbol(:Δz, L1, :ᵃ, L2)
    Δzᵃˡˡ = Symbol(:Δz, :ᵃ, L1, L2)

    # 2D areas
    Axᵃˡˡ = Symbol(:Ax, :ᵃ, L1, L2)
    Ayˡᵃˡ = Symbol(:Ay, L1, :ᵃ, L2)
    Azˡˡᵃ = Symbol(:Az, L1, L2, :ᵃ)

    @eval begin
        @inline $Axᵃˡˡ(i, j, k, grid) = $Δyᵃˡˡ(i, j, k, grid) * $Δzᵃˡˡ(i, j, k, grid)
        @inline $Ayˡᵃˡ(i, j, k, grid) = $Δxˡᵃˡ(i, j, k, grid) * $Δzˡᵃˡ(i, j, k, grid)
        @inline $Azˡˡᵃ(i, j, k, grid) = $Δxˡˡᵃ(i, j, k, grid) * $Δyˡˡᵃ(i, j, k, grid)
    end

    for L3 in (:ᶜ, :ᶠ)
        # 3D spacings
        Δxˡˡˡ = Symbol(:Δx, L1, L2, L3)
        Δyˡˡˡ = Symbol(:Δy, L1, L2, L3)
        Δzˡˡˡ = Symbol(:Δz, L1, L2, L3)

        # 3D areas
        Axˡˡˡ = Symbol(:Ax, L1, L2, L3)
        Ayˡˡˡ = Symbol(:Ay, L1, L2, L3)
        Azˡˡˡ = Symbol(:Az, L1, L2, L3)

        @eval begin
            @inline $Axˡˡˡ(i, j, k, grid) = $Δyˡˡˡ(i, j, k, grid) * $Δzˡˡˡ(i, j, k, grid)
            @inline $Ayˡˡˡ(i, j, k, grid) = $Δxˡˡˡ(i, j, k, grid) * $Δzˡˡˡ(i, j, k, grid)

            # For the moment the horizontal area is independent of `z`. This might change if
            # we want to implement deep atmospheres where Az is a function of z
            @inline $Azˡˡˡ(i, j, k, grid) = $Azˡˡᵃ(i, j, k, grid)
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

@inline Azᶠᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))

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
##### Volumes!! (always 3D)
#####
#####

for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)

    volume    = Symbol(:V,  LX, LY, LZ)
    z_area    = Symbol(:Az, LX, LY, LZ)
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

            # General spacing functions
            for dir in (:x, :y, :λ, :φ, :z, :r)
                func   = Symbol(:Δ, dir)
                metric = Symbol(:Δ, dir, location_code(LXe, LYe, LZe))
                rcp_func   = Symbol(:Δ, dir, :⁻¹)
                rcp_metric = Symbol(:Δ, dir, :⁻¹, location_code(LXe, LYe, LZe))

                @eval begin
                    @inline $func(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $metric(i, j, k, grid)
                    @inline $rcp_func(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $rcp_metric(i, j, k, grid)
                    export $metric, $rcp_metric
                end
            end

            # General area functions
            for dir in (:x, :y, :z)
                func   = Symbol(:A, dir)
                metric = Symbol(:A, dir, location_code(LXe, LYe, LZe))
                rcp_func   = Symbol(:A, dir, :⁻¹)
                rcp_metric = Symbol(:A, dir, :⁻¹, location_code(LXe, LYe, LZe))

                @eval begin
                    @inline $func(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $metric(i, j, k, grid)
                    @inline $rcp_func(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $rcp_metric(i, j, k, grid)
                    export $metric, $rcp_metric
                end
            end

            # General volume function
            volume_function = Symbol(:V, location_code(LXe, LYe, LZe))
            rcp_volume_function = Symbol(:V⁻¹, location_code(LXe, LYe, LZe))

            @eval begin
                @inline volume(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $volume_function(i, j, k, grid)
                @inline rcp_volume(i, j, k, grid, ::$LX, ::$LY, ::$LZ) = $volume_function(i, j, k, grid)
                export $volume_function, $rcp_volume_function
            end
        end
    end
end

# One-dimensional convenience spacings (for grids that support them)

Δx(i, grid, ℓx) = Δx(i, 1, 1, grid, ℓx, nothing, nothing)
Δy(j, grid, ℓy) = Δy(1, j, 1, grid, nothing, ℓy, nothing)
Δz(k, grid, ℓz) = Δz(1, 1, k, grid, nothing, nothing, ℓz)
Δλ(i, grid, ℓx) = Δλ(i, 1, 1, grid, ℓx, nothing, nothing)
Δφ(j, grid, ℓy) = Δφ(1, j, 1, grid, nothing, ℓy, nothing)
Δr(k, grid, ℓz) = Δr(1, 1, k, grid, nothing, nothing, ℓz)

# Two-dimensional horizontal convenience spacings (for grids that support them)

Δx(i, j, grid, ℓx, ℓy) = Δx(i, j, 1, grid, ℓx, ℓy, nothing)
Δy(i, j, grid, ℓx, ℓy) = Δy(i, j, 1, grid, ℓx, ℓy, nothing)
Δλ(i, j, grid, ℓx, ℓy) = Δλ(i, j, 1, grid, ℓx, ℓy, nothing)
Δφ(i, j, grid, ℓx, ℓy) = Δφ(i, j, 1, grid, ℓx, ℓy, nothing)
