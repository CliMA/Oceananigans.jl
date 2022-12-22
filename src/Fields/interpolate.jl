using Oceananigans.Grids: XRegRectilinearGrid, YRegRectilinearGrid, ZRegRectilinearGrid
using Oceananigans.Grids: XRegLatLonGrid, YRegLatLonGrid, ZRegLatLonGrid
using Oceananigans.Grids: xnodes, ynodes, znodes, topology

# GPU-compatile middle point calculation
@inline middle_point(l, h) = Base.unsafe_trunc(Int, (l + h) / 2)

"""
    index_binary_search(vec, val, array_size)

Return indices `low, high` of `vec`tor for which 

```
vec[low] <= val && vec[high] >= val
```

using a binary search. The input array `vec` has to be monotonically increasing.

Code credit: https://computersciencehub.io/julia/code-for-binary-search-algorithm-julia
"""
@inline function index_binary_search(vec, val, array_size)
    low = 0
    high = array_size - 1

    while low + 1 < high 
        mid = middle_point(low, high)
        if @inbounds vec[mid + 1] == val 
            return (mid + 1, mid + 1)
        elseif @inbounds vec[mid + 1] < val
            low = mid
        else
            high = mid
        end
    end

    return (low + 1, high + 1)
end

@inline function fractional_index(array_size::Int, val::FT, vec) where {FT}
    y₁, y₂ = index_binary_search(vec, val, array_size)

    @inbounds x₂ = vec[y₂]
    @inbounds x₁ = vec[y₁]

    if y₁ == y₂
        return FT(y₁)
    else
        return FT((y₂ - y₁) / (x₂ - x₁) * (val - x₁) + y₁)
    end
end

####
#### Disclaimer! interpolation on LatitudeLongitudeGrid assumes a thin shell (i.e. no curvature effects when interpolating)
#### Use other methods if a more accurate interpolation is required
####

@inline fractional_x_index(x::FT, ::Center, grid) where FT = fractional_index(length(Center, topology(grid)[1], grid.Nx), x, xnodes(Center, grid))
@inline fractional_y_index(y::FT, ::Center, grid) where FT = fractional_index(length(Center, topology(grid)[2], grid.Ny), y, ynodes(Center, grid))
@inline fractional_z_index(z::FT, ::Center, grid) where FT = fractional_index(length(Center, topology(grid)[3], grid.Nz), z, znodes(Center, grid))

@inline fractional_x_index(x::FT, ::Face, grid) where FT = fractional_index(length(Face, topology(grid)[1], grid.Nx), x, xnodes(Face, grid)) - 1
@inline fractional_y_index(y::FT, ::Face, grid) where FT = fractional_index(length(Face, topology(grid)[2], grid.Ny), y, ynodes(Face, grid)) - 1
@inline fractional_z_index(z::FT, ::Face, grid) where FT = fractional_index(length(Face, topology(grid)[3], grid.Nz), z, znodes(Face, grid)) - 1

@inline fractional_x_index(x::FT, ::Face,   grid::XRegRectilinearGrid) where FT = @inbounds FT((x - grid.xᶠᵃᵃ[1]) / grid.Δxᶠᵃᵃ)
@inline fractional_x_index(x::FT, ::Center, grid::XRegRectilinearGrid) where FT = @inbounds FT((x - grid.xᶜᵃᵃ[1]) / grid.Δxᶜᵃᵃ)
@inline fractional_y_index(y::FT, ::Face,   grid::YRegRectilinearGrid) where FT = @inbounds FT((y - grid.yᵃᶠᵃ[1]) / grid.Δyᵃᶠᵃ)
@inline fractional_y_index(y::FT, ::Center, grid::YRegRectilinearGrid) where FT = @inbounds FT((y - grid.yᵃᶜᵃ[1]) / grid.Δyᵃᶜᵃ)

@inline fractional_x_index(λ::FT, ::Face,   grid::XRegLatLonGrid) where FT = @inbounds FT((λ - grid.λᶠᵃᵃ[1]) / grid.Δλᶠᵃᵃ)
@inline fractional_x_index(λ::FT, ::Center, grid::XRegLatLonGrid) where FT = @inbounds FT((λ - grid.λᶜᵃᵃ[1]) / grid.Δλᶜᵃᵃ)
@inline fractional_y_index(φ::FT, ::Face,   grid::YRegLatLonGrid) where FT = @inbounds FT((φ - grid.φᵃᶠᵃ[1]) / grid.Δφᵃᶠᵃ)
@inline fractional_y_index(φ::FT, ::Center, grid::YRegLatLonGrid) where FT = @inbounds FT((φ - grid.φᵃᶜᵃ[1]) / grid.Δφᵃᶜᵃ)

const ZReg = Union{ZRegRectilinearGrid, ZRegLatLonGrid}

@inline fractional_z_index(z, ::Face,   grid::ZReg) = @inbounds (z - grid.zᵃᵃᶠ[1]) / grid.Δzᵃᵃᶠ
@inline fractional_z_index(z, ::Center, grid::ZReg) = @inbounds (z - grid.zᵃᵃᶜ[1]) / grid.Δzᵃᵃᶜ

"""
    fractional_indices(x, y, z, loc, grid)

Convert the coordinates `(x, y, z)` to _fractional_ indices on a regular rectilinear grid located at `loc`
where `loc` is a 3-tuple of `Center` and `Face`. Fractional indices are floats indicating a location between
grid points.
"""
@inline function fractional_indices(x, y, z, loc, grid)
    i = fractional_x_index(x, loc[1], grid)
    j = fractional_y_index(y, loc[2], grid)
    k = fractional_z_index(z, loc[3], grid)
    
    return (i, j, k)
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
