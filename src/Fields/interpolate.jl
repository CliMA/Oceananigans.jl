using Oceananigans.Grids: XRegRectilinearGrid, YRegRectilinearGrid, ZRegRectilinearGrid
using CUDA: allowscalar 

@inline function linear_interpolate_sorted_vector(vec, val)

    allowscalar(true)
    y2 = searchsortedfirst(vec, val)
    y1 = searchsortedlast(vec, val)
    x2 = vec[y2]
    x1 = vec[y1]
    allowscalar(false)

    return (y2 - y1) / (x2 - x1) * (val - x1) + y1
end

####
#### Disclaimer! interpolation on LatitudeLongitudeGrid assumes a thin shell (i.e. no curvature effects when interpolating)
#### Use other methods if a more accurate interpolation is required
####

@inline fractional_x_index(x, ::Face,   grid::RectilinearGrid)     = linear_interpolate_sorted_vector(grid.xᶠᵃᵃ, x)
@inline fractional_x_index(x, ::Center, grid::RectilinearGrid)     = linear_interpolate_sorted_vector(grid.xᶜᵃᵃ, x)
@inline fractional_y_index(y, ::Face,   grid::RectilinearGrid)     = linear_interpolate_sorted_vector(grid.yᵃᶠᵃ, y)
@inline fractional_y_index(y, ::Center, grid::RectilinearGrid)     = linear_interpolate_sorted_vector(grid.yᵃᶜᵃ, y)

@inline fractional_x_index(x, ::Face,   grid::XRegRectilinearGrid) = @inbounds (x - grid.xᶠᵃᵃ[1]) / grid.Δxᶠᵃᵃ
@inline fractional_x_index(x, ::Center, grid::XRegRectilinearGrid) = @inbounds (x - grid.xᶜᵃᵃ[1]) / grid.Δxᶜᵃᵃ
@inline fractional_y_index(y, ::Face,   grid::YRegRectilinearGrid) = @inbounds (y - grid.yᵃᶠᵃ[1]) / grid.Δyᵃᶠᵃ
@inline fractional_y_index(y, ::Center, grid::YRegRectilinearGrid) = @inbounds (y - grid.yᵃᶜᵃ[1]) / grid.Δyᵃᶜᵃ

@inline fractional_x_index(λ, ::Face,   grid::LatitudeLongitudeGrid) = linear_interpolate_sorted_vector(grid.λᶠᵃᵃ, λ)
@inline fractional_x_index(λ, ::Center, grid::LatitudeLongitudeGrid) = linear_interpolate_sorted_vector(grid.λᶜᵃᵃ, λ)
@inline fractional_y_index(φ, ::Face,   grid::LatitudeLongitudeGrid) = linear_interpolate_sorted_vector(grid.φᵃᶠᵃ, φ)
@inline fractional_y_index(φ, ::Center, grid::LatitudeLongitudeGrid) = linear_interpolate_sorted_vector(grid.φᵃᶜᵃ, φ)

@inline fractional_x_index(λ, ::Face,   grid::XRegLatLonGrid) = @inbounds (λ - grid.λᶠᵃᵃ[1]) / grid.Δλᶠᵃᵃ
@inline fractional_x_index(λ, ::Center, grid::XRegLatLonGrid) = @inbounds (λ - grid.λᶜᵃᵃ[1]) / grid.Δλᶜᵃᵃ
@inline fractional_y_index(φ, ::Face,   grid::YRegLatLonGrid) = @inbounds (φ - grid.φᵃᶠᵃ[1]) / grid.Δφᵃᶠᵃ
@inline fractional_y_index(φ, ::Center, grid::YRegLatLonGrid) = @inbounds (φ - grid.φᵃᶜᵃ[1]) / grid.Δφᵃᶜᵃ

const ZReg = Union{ZRegRectilinearGrid, ZRegLatLonGrid}

@inline fractional_z_index(z, ::Face,   grid) = linear_interpolate_sorted_vector(grid.zᵃᵃᶠ, z)
@inline fractional_z_index(z, ::Center, grid) = linear_interpolate_sorted_vector(grid.zᵃᵃᶜ, z)

@inline fractional_z_index(z, ::Face,   grid::ZReg) = @inbounds (z - grid.zᵃᵃᶠ[1]) / grid.Δzᵃᵃᶠ
@inline fractional_z_index(z, ::Center, grid::ZReg) = @inbounds (z - grid.zᵃᵃᶜ[1]) / grid.Δzᵃᵃᶜ

"""
    fractional_indices(x, y, z, loc, grid::RectilinearGrid)

Convert the coordinates `(x, y, z)` to _fractional_ indices on a regular rectilinear grid located at `loc`
where `loc` is a 3-tuple of `Center` and `Face`. Fractional indices are floats indicating a location between
grid points.
"""
@inline function fractional_indices(x, y, z, loc, grid)
    i = fractional_x_index(x, loc[1], grid)
    j = fractional_y_index(y, loc[2], grid)
    k = fractional_z_index(z, loc[3], grid)
    return i, j, k
end

# Trilinear Lagrange polynomials
@inline ϕ₁(ξ, η, ζ) = (1 - ξ) * (1 - η) * (1 - ζ)
@inline ϕ₂(ξ, η, ζ) = (1 - ξ) * (1 - η) *      ζ
@inline ϕ₃(ξ, η, ζ) = (1 - ξ) *      η  * (1 - ζ)
@inline ϕ₄(ξ, η, ζ) = (1 - ξ) *      η  *      ζ
@inline ϕ₅(ξ, η, ζ) =      ξ  * (1 - η) * (1 - ζ)
@inline ϕ₆(ξ, η, ζ) =      ξ  * (1 - η) *      ζ
@inline ϕ₇(ξ, η, ζ) =      ξ  *      η  * (1 - ζ)
@inline ϕ₈(ξ, η, ζ) =      ξ  *      η  *      ζ

@inline _interpolate(field, ξ, η, ζ, i, j, k) =
    @inbounds (  ϕ₁(ξ, η, ζ) * field[i,   j,   k  ]
               + ϕ₂(ξ, η, ζ) * field[i,   j,   k+1]
               + ϕ₃(ξ, η, ζ) * field[i,   j+1, k  ]
               + ϕ₄(ξ, η, ζ) * field[i,   j+1, k+1]
               + ϕ₅(ξ, η, ζ) * field[i+1, j,   k  ]
               + ϕ₆(ξ, η, ζ) * field[i+1, j,   k+1]
               + ϕ₇(ξ, η, ζ) * field[i+1, j+1, k  ]
               + ϕ₈(ξ, η, ζ) * field[i+1, j+1, k+1])

"""
    interpolate(field, x, y, z)

Interpolate `field` to the physical point `(x, y, z)` using trilinear interpolation.
"""
@inline function interpolate(field, x, y, z)
    LX, LY, LZ = location(field)
    i, j, k = fractional_indices(x, y, z, (LX(), LY(), LZ()), field.grid)

    # Convert fractional indices to unit cell coordinates 0 <= (ξ, η, ζ) <=1
    # and integer indices (with 0-based indexing).
    ξ, i = modf(i)
    η, j = modf(j)
    ζ, k = modf(k)

    # Convert indices to proper integers and shift to 1-based indexing.
    return _interpolate(field, ξ, η, ζ, Int(i+1), Int(j+1), Int(k+1))
end

"""
    interpolate(field, LX, LY, LZ, grid, x, y, z)

Interpolate `field` to the physical point `(x, y, z)` using trilinear interpolation. The location of
the field is specified with `(LX, LY, LZ)` and the field is defined on `grid`.

Note that this is a lower-level `interpolate` method defined for use in CPU/GPU kernels.
"""

@inline function interpolate(field, LX, LY, LZ, grid, x, y, z)
    i, j, k = fractional_indices(x, y, z, (LX, LY, LZ), grid)

    # We use mod and trunc as CUDA.modf is not defined.
    # For why we use Base.unsafe_trunc instead of trunc see:
    # https://github.com/CliMA/Oceananigans.jl/issues/828
    # https://github.com/CliMA/Oceananigans.jl/pull/997
    ξ, i = mod(i, 1), Base.unsafe_trunc(Int, i)
    η, j = mod(j, 1), Base.unsafe_trunc(Int, j)
    ζ, k = mod(k, 1), Base.unsafe_trunc(Int, k)

    return _interpolate(field, ξ, η, ζ, i+1, j+1, k+1)
end
