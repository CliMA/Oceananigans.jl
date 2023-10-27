using Oceananigans.Grids: topology, node,
                          xspacings, yspacings, zspacings, λspacings, φspacings,
                          XFlatGrid, YFlatGrid, ZFlatGrid,
                          XRegularRG, YRegularRG, ZRegularRG,
                          XRegularLLG, YRegularLLG, ZRegularLLG,
                          ZRegOrthogonalSphericalShellGrid,
                          RectilinearGrid, LatitudeLongitudeGrid

# GPU-compatile middle point calculation
@inline middle_point(l, h) = Base.unsafe_trunc(Int, (l + h) / 2)

"""
    index_binary_search(vec, val, array_size)

Return indices `low, high` of `vec`tor for which 

```julia
vec[low] ≤ val && vec[high] ≥ val
```

using a binary search. The input array `vec` has to be monotonically increasing.

Code credit: https://gist.github.com/cuongld2/8e4fed9ba44ea2b4598f90e7d5b6c612/155f9cb595314c8db3a266c3316889443b068017
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

@inline function fractional_index(array_size::Int, val::FT, vec) where FT
    y₁, y₂ = index_binary_search(vec, val, array_size)

    @inbounds x₁ = vec[y₁]
    @inbounds x₂ = vec[y₂]

    if y₁ == y₂
        return convert(FT, y₁)
    else
        return convert(FT, (y₂ - y₁) / (x₂ - x₁) * (val - x₁) + y₁)
    end
end

#####
##### Disclaimer! interpolation on LatitudeLongitudeGrid assumes a thin shell (i.e. no curvature effects when interpolating)
##### Use other methods if a more accurate interpolation is required
#####

@inline fractional_x_index(x, locs, grid::XFlatGrid) = zero(grid)

@inline function fractional_x_index(x::FT, locs, grid::XRegularRG) where FT
    x₀ = @inbounds node(1, 1, 1, grid, locs...)[1]
    Δx = xspacings(grid, locs...)
    return convert(FT, (x - x₀) / Δx)
end

@inline function fractional_x_index(λ::FT, locs, grid::XRegularLLG) where FT
    λ₀ = @inbounds node(1, 1, 1, grid, locs...)[1]
    Δλ = λspacings(grid, locs...)
    return convert(FT, (λ - λ₀) / Δλ)
end

@inline function fractional_x_index(x, locs, grid)
    loc = @inbounds locs[1]
     Tx = topology(grid, 1)()
      L = length(loc, Tx, grid.Nx)
     xn = @inbounds nodes(grid, locs)[1]
    return fractional_index(L, x, xn) - 1
end

@inline fractional_y_index(y, locs, grid::YFlatGrid) = zero(grid)

@inline function fractional_y_index(y::FT, locs, grid::YRegularRG) where FT
    y₀ = @inbounds node(1, 1, 1, grid, locs...)[2]
    Δy = yspacings(grid, locs...)
    return convert(FT, (y - y₀) / Δy)
end

@inline function fractional_y_index(φ::FT, locs, grid::YRegularLLG) where FT
    φ₀ = @inbounds node(1, 1, 1, grid, locs...)[2]
    Δφ = φspacings(grid, locs...)
    return convert(FT, (φ - φ₀) / Δφ)
end

@inline function fractional_y_index(y, locs, grid)
    loc = @inbounds locs[2]
     Ty = topology(grid, 2)()
      L = length(loc, Ty, grid.Ny)
     yn = nodes(grid, locs)[2]
    return fractional_index(L, y, yn) - 1
end

@inline fractional_z_index(z, locs, grid::ZFlatGrid) = zero(grid)

ZRegGrid = Union{ZRegularRG, ZRegularLLG, ZRegOrthogonalSphericalShellGrid}

@inline function fractional_z_index(z::FT, locs, grid::ZRegGrid) where FT
    z₀ = @inbounds node(1, 1, 1, grid, locs...)[3]
    Δz = zspacings(grid, locs...)
    return convert(FT, (z - z₀) / Δz)
end

@inline function fractional_z_index(z, locs, grid)
    loc = @inbounds locs[3]
     Tz = topology(grid, 3)()
      L = length(loc, Tz, grid.Nz)
     zn = znodes(grid, loc)
    return fractional_index(L, z, zn) - 1
end

"""
    fractional_indices(x, y, z, loc, grid)

Convert the coordinates `(x, y, z)` to _fractional_ indices on a regular rectilinear grid
located at `loc`, where `loc` is a 3-tuple of `Center` and `Face`. Fractional indices are
floats indicating a location between grid points.
"""
@inline function fractional_indices(x, y, z, locs, grid)
    i = fractional_x_index(x, locs, grid)
    j = fractional_y_index(y, locs, grid)
    k = fractional_z_index(z, locs, grid)

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
