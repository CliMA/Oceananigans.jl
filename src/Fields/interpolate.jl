using Oceananigans.Grids: topology, node,
                          xspacings, yspacings, zspacings, λspacings, φspacings,
                          XFlatGrid, YFlatGrid, ZFlatGrid,
                          XRegRectilinearGrid, YRegRectilinearGrid, ZRegRectilinearGrid,
                          XRegLatLonGrid, YRegLatLonGrid, ZRegLatLonGrid,
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

@inline function fractional_x_index(x::FT, locs, grid::XRegRectilinearGrid) where FT
    x₀ = @inbounds xnode(1, 1, 1, grid, locs...)
    Δx = @inbounds xspacings(grid, locs...)
    return convert(FT, (x - x₀) / Δx)
end

@inline function fractional_x_index(λ::FT, locs, grid::XRegLatLonGrid) where FT
    λ₀ = @inbounds λnode(1, 1, 1, grid, locs...)
    Δλ = @inbounds λspacings(grid, locs...)
    return convert(FT, (λ - λ₀) / Δλ)
end

@inline function fractional_x_index(x, locs, grid)
    loc = @inbounds locs[1]
     Tx = topology(grid, 1)()
      L = length(loc, Tx, grid.Nx)
     xn = @inbounds xnodes(grid, locs)
    return fractional_index(L, x, xn) - 1
end

@inline fractional_y_index(y, locs, grid::YFlatGrid) = zero(grid)

@inline function fractional_y_index(y::FT, locs, grid::YRegRectilinearGrid) where FT
    y₀ = @inbounds ynode(1, 1, 1, grid, locs...)
    Δy = @inbounds yspacings(grid, locs...)
    return convert(FT, (y - y₀) / Δy)
end

@inline function fractional_y_index(φ::FT, locs, grid::YRegLatLonGrid) where FT
    φ₀ = @inbounds φnode(1, 1, 1, grid, locs...)
    Δφ = @inbounds φspacings(grid, locs...)
    return convert(FT, (φ - φ₀) / Δφ)
end

@inline function fractional_y_index(y, locs, grid)
    loc = @inbounds locs[2]
     Ty = topology(grid, 2)()
      L = length(loc, Ty, grid.Ny)
     yn = ynodes(grid, locs)
    return fractional_index(L, y, yn) - 1
end

@inline fractional_z_index(z, locs, grid::ZFlatGrid) = zero(grid)

ZRegGrid = Union{ZRegRectilinearGrid, ZRegLatLonGrid, ZRegOrthogonalSphericalShellGrid}

@inline function fractional_z_index(z::FT, locs, grid::ZRegGrid) where FT
    z₀ = @inbounds znode(1, 1, 1, grid, locs...)
    Δz = @inbounds zspacings(grid, locs...)
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
@inline fractional_indices(at_node, grid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, ℓx, ℓy, ℓz)

@inline fractional_indices(at_node, grid::XFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, nothing, ℓy, ℓz)
@inline fractional_indices(at_node, grid::YFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, ℓx, nothing, ℓz)
@inline fractional_indices(at_node, grid::ZFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, ℓx, ℓy, nothing)

@inline fractional_indices(at_node, grid::XYFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, nothing, nothing, ℓz)
@inline fractional_indices(at_node, grid::YZFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, ℓx, nothing, nothing)
@inline fractional_indices(at_node, grid::XZFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, nothing, ℓy, nothing)

@inline function _fractional_indices((x, y, z), grid, ℓx, ℓy, ℓz)
    i = fractional_x_index(x, locs, grid)
    j = fractional_y_index(y, locs, grid)
    k = fractional_z_index(z, locs, grid)
    return (i, j, k)
end

@inline function _fractional_indices((y, z), grid, ::Nothing, ℓy, ℓz)
    j = fractional_y_index(y, (nothing, ℓy, ℓz), grid)
    k = fractional_z_index(z, (nothing, ℓy, ℓz), grid)
    return (nothing, j, k)
end

@inline function _fractional_indices((x, z), grid, ℓx, ::Nothing, ℓz)
    i = fractional_x_index(x, (ℓx, nothing, ℓz), grid)
    k = fractional_z_index(z, (ℓx, nothing, ℓz), grid)
    return (i, nothing, k)
end

@inline function _fractional_indices((x, y), grid, ℓx, ℓy, ::Nothing)
    i = fractional_x_index(x, (ℓx, ℓy, nothing), grid)
    j = fractional_y_index(y, (ℓx, ℓy, nothing), grid)
    return (i, j, nothing)
end

@inline function _fractional_indices((x,), grid, ℓx, ::Nothing, ::Nothing)
    loc = (ℓx, nothing, nothing)
    ii = fractional_x_index(x, loc, grid)
    jj = nothing
    kk = nothing
    return (ii, jj, kk)
end

@inline function _fractional_indices((y,), grid, ::Nothing, ℓy, ::Nothing)
    loc = (nothing, ℓy, nothing)
    ii = nothing
    jj = fractional_y_index(y, loc, grid)
    kk = nothing
    return (ii, jj, kk)
end

@inline function _fractional_indices((z,), grid, ::Nothing, ::Nothing, ℓz)
    loc = (nothing, nothing, ℓz)
    ii = nothing
    jj = nothing
    kk = fractional_z_index(z, loc, grid)
    return (ii, jj, kk)
end

"""
    interpolate(field, LX, LY, LZ, grid, x, y, z)

Interpolate `field` to the physical point `(x, y, z)` using trilinear interpolation. The location of
the field is specified with `(LX, LY, LZ)` and the field is defined on `grid`.

Note that this is a lower-level `interpolate` method defined for use in CPU/GPU kernels.
"""
@inline function interpolate(to_node, from_field, from_loc, from_grid)
    # field, LX, LY, LZ, grid, x, y, z)
    ii, jj, kk = fractional_indices(at_node, from_grid, from_loc...)

    # We use mod and trunc as CUDA.modf is not defined.
    # For why we use Base.unsafe_trunc instead of trunc see:
    # https://github.com/CliMA/Oceananigans.jl/issues/828
    # https://github.com/CliMA/Oceananigans.jl/pull/997
    ix = interpolator(ii)
    iy = interpolator(jj)
    iz = interpolator(kk)

    return _interpolate(field, ix, iy, iz)
end

"""
    interpolator(ii)

Given the ``interpolator tuple'' for the ``fractional'' index `ii`
defined as the 3-tuple

```
(i⁻, i⁺, ξ)
```

where `i⁻` is the index to the left of `ii`, `i⁺` is the index to the
right of `ii`, and `ξ` is the fractional distance between `ii` and the
left bound `i⁻`, such that `ξ ∈ [0, 1)`.
"""
@inline function interpolator(fractional_idx)
    i⁻ = Base.unsafe_trunc(Int, fractional_idx) + 1
    i⁺ = i⁻ + 1
    ξ = mod(fractional_idx, 1)
    return i⁻, i⁺, ξ
end

@inline interpolator(::Nothing) = (1, 1, 0)

# Trilinear Lagrange polynomials
@inline ϕ₁(ξ, η, ζ) = (1 - ξ) * (1 - η) * (1 - ζ)
@inline ϕ₂(ξ, η, ζ) = (1 - ξ) * (1 - η) *      ζ
@inline ϕ₃(ξ, η, ζ) = (1 - ξ) *      η  * (1 - ζ)
@inline ϕ₄(ξ, η, ζ) = (1 - ξ) *      η  *      ζ
@inline ϕ₅(ξ, η, ζ) =      ξ  * (1 - η) * (1 - ζ)
@inline ϕ₆(ξ, η, ζ) =      ξ  * (1 - η) *      ζ
@inline ϕ₇(ξ, η, ζ) =      ξ  *      η  * (1 - ζ)
@inline ϕ₈(ξ, η, ζ) =      ξ  *      η  *      ζ

@inline function _interpolate(data, ix, iy, iz)
    # Unpack the "interpolators"
    i⁻, i⁺, ξ = ix
    j⁻, j⁺, η = iy
    k⁻, k⁺, ζ = iz

    return @inbounds ϕ₁(ξ, η, ζ) * data[i⁻, j⁻, k⁻] +
                     ϕ₂(ξ, η, ζ) * data[i⁻, j⁻, k⁺] +  
                     ϕ₃(ξ, η, ζ) * data[i⁻, j⁺, k⁻] +
                     ϕ₄(ξ, η, ζ) * data[i⁻, j⁺, k⁺] +
                     ϕ₅(ξ, η, ζ) * data[i⁺, j⁻, k⁻] +
                     ϕ₆(ξ, η, ζ) * data[i⁺, j⁻, k⁺] +
                     ϕ₇(ξ, η, ζ) * data[i⁺, j⁺, k⁻] +
                     ϕ₈(ξ, η, ζ) * data[i⁺, j⁺, k⁺]
end

"""
    interpolate(field, x, y, z)

Interpolate `field` to the physical point `(x, y, z)` using trilinear interpolation.
"""
@inline function interpolate(to_node, from_field)
    from_loc = Tuple(L() for L in location(from_field))
    return interpolate(to_node, from_field, from_loc, from_field.grid)
end


