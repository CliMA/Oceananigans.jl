using Oceananigans.Grids: topology, node,
                          xspacings, yspacings, zspacings, λspacings, φspacings,
                          XFlatGrid, YFlatGrid, ZFlatGrid,
                          XYFlatGrid, YZFlatGrid, XZFlatGrid,
                          XRegularRG, YRegularRG, ZRegularRG,
                          XRegularLLG, YRegularLLG, ZRegularLLG,
                          ZRegOrthogonalSphericalShellGrid,
                          RectilinearGrid, LatitudeLongitudeGrid

# GPU-compatile middle point calculation
@inline middle_point(l, h) = Base.unsafe_trunc(Int, (l + h) / 2)

"""
    index_binary_search(val, vec, N)

Return indices `low, high` of `vec`tor for which 

```julia
vec[low] ≤ val && vec[high] ≥ val
```

using a binary search. The input array `vec` has to be monotonically increasing.

Code credit: https://gist.github.com/cuongld2/8e4fed9ba44ea2b4598f90e7d5b6c612/155f9cb595314c8db3a266c3316889443b068017
"""
@inline function index_binary_search(val, vec, N)
    low = 0
    high = N - 1

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

@inline function fractional_index(val, vec, N)
    i₁, i₂ = index_binary_search(val, vec, N)

    @inbounds x₁ = vec[i₁]
    @inbounds x₂ = vec[i₂]

    ii = (i₂ - i₁) / (x₂ - x₁) * (val - x₁) + i₁
    ii = ifelse(i₁ == i₂, i₁, ii)

    FT = typeof(val) # convert "fractional index" to type of `val`
    return convert(FT, ii)
end

#####
##### Disclaimer! interpolation on LatitudeLongitudeGrid assumes a thin shell (i.e. no curvature effects when interpolating)
##### Use other methods if a more accurate interpolation is required
#####

@inline fractional_x_index(x, locs, grid::XFlatGrid) = zero(grid)

@inline function fractional_x_index(x, locs, grid::XRegularRG)
    x₀ = xnode(1, 1, 1, grid, locs...)
    Δx = xspacings(grid, locs...)
    FT = eltype(grid)
    return convert(FT, (x - x₀) / Δx)
end

@inline function fractional_x_index(λ, locs, grid::XRegularLLG)
    λ₀ = @inbounds λnode(1, 1, 1, grid, locs...)
    Δλ = λspacings(grid, locs...)
    FT = eltype(grid)
    return convert(FT, (λ - λ₀) / Δλ)
end

@inline function fractional_x_index(x, locs, grid::RectilinearGrid)
    loc = @inbounds locs[1]
     Tx = topology(grid, 1)()
     Nx = length(loc, Tx, grid.Nx)
     xn = xnodes(grid, locs...)
    return fractional_index(x, xn, Nx) - 1
end

@inline function fractional_x_index(x, locs, grid::LatitudeLongitudeGrid)
    loc = @inbounds locs[1]
     Tx = topology(grid, 1)()
     Nx = length(loc, Tx, grid.Nx)
     xn = λnodes(grid, locs...)
    return fractional_index(x, xn, Nx) - 1
end

@inline fractional_y_index(y, locs, grid::YFlatGrid) = zero(grid)

@inline function fractional_y_index(y, locs, grid::YRegularRG)
    y₀ = @inbounds ynode(1, 1, 1, grid, locs...)
    Δy = @inbounds yspacings(grid, locs...)
    FT = eltype(grid)
    return convert(FT, (y - y₀) / Δy)
end

@inline function fractional_y_index(φ, locs, grid::YRegularLLG)
    φ₀ = @inbounds φnode(1, 1, 1, grid, locs...)
    Δφ = @inbounds φspacings(grid, locs...)
    FT = eltype(grid)
    return convert(FT, (φ - φ₀) / Δφ)
end

@inline function fractional_y_index(y, locs, grid::RectilinearGrid)
    loc = @inbounds locs[2]
     Ty = topology(grid, 2)()
     Ny = length(loc, Ty, grid.Ny)
     yn = ynodes(grid, locs...)
    return fractional_index(y, yn, Ny) - 1
end

@inline function fractional_y_index(y, locs, grid::LatitudeLongitudeGrid)
    loc = @inbounds locs[2]
     Ty = topology(grid, 2)()
     Ny = length(loc, Ty, grid.Ny)
     yn = φnodes(grid, locs...)
    return fractional_index(y, yn, Ny) - 1
end

@inline fractional_z_index(z, locs, grid::ZFlatGrid) = zero(grid)

ZRegGrid = Union{ZRegularRG, ZRegularLLG, ZRegOrthogonalSphericalShellGrid}

@inline function fractional_z_index(z::FT, locs, grid::ZRegGrid) where FT
    z₀ = @inbounds znode(1, 1, 1, grid, locs...)
    Δz = @inbounds zspacings(grid, locs...)
    return convert(FT, (z - z₀) / Δz)
end

@inline function fractional_z_index(z, locs, grid)
    loc = @inbounds locs[3]
     Tz = topology(grid, 3)()
     Nz = length(loc, Tz, grid.Nz)
     zn = znodes(grid, loc)
    return fractional_index(z, zn, Nz) - 1
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
    ii = fractional_x_index(x, (ℓx, ℓy, ℓz), grid)
    jj = fractional_y_index(y, (ℓx, ℓy, ℓz), grid)
    kk = fractional_z_index(z, (ℓx, ℓy, ℓz), grid)
    return (ii, jj, kk)
end

@inline function _fractional_indices((y, z), grid, ::Nothing, ℓy, ℓz)
    jj = fractional_y_index(y, (nothing, ℓy, ℓz), grid)
    kk = fractional_z_index(z, (nothing, ℓy, ℓz), grid)
    return (nothing, jj, kk)
end

@inline function _fractional_indices((x, z), grid, ℓx, ::Nothing, ℓz)
    ii = fractional_x_index(x, (ℓx, nothing, ℓz), grid)
    kk = fractional_z_index(z, (ℓx, nothing, ℓz), grid)
    return (ii, nothing, kk)
end

@inline function _fractional_indices((x, y), grid, ℓx, ℓy, ::Nothing)
    ii = fractional_x_index(x, (ℓx, ℓy, nothing), grid)
    jj = fractional_y_index(y, (ℓx, ℓy, nothing), grid)

    return (ii, jj, nothing)
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
    interpolate(at_node, from_field, from_loc, from_grid)

Interpolate `from_field`, `at_node`, on `from_grid` and at `from_loc`ation,
where `at_node` is a tuple of coordinates and and `from_loc = (ℓx, ℓy, ℓz)`.

Note that this is a lower-level `interpolate` method defined for use in CPU/GPU kernels.
"""
@inline function interpolate(at_node, from_field, from_loc, from_grid)
    # field, LX, LY, LZ, grid, x, y, z)
    ii, jj, kk = fractional_indices(at_node, from_grid, from_loc...)

    ix = interpolator(ii)
    iy = interpolator(jj)
    iz = interpolator(kk)

    return _interpolate(from_field, ix, iy, iz)
end

"""
    interpolator(ii)

Return an ``interpolator tuple'' from the ``fractional'' index `ii`
defined as the 3-tuple

```
(i⁻, i⁺, ξ)
```

where `i⁻` is the index to the left of `ii`, `i⁺` is the index to the
right of `ii`, and `ξ` is the fractional distance between `ii` and the
left bound `i⁻`, such that `ξ ∈ [0, 1)`.
"""
@inline function interpolator(fractional_idx)
    # We use mod and trunc as CUDA.modf is not defined.
    # For why we use Base.unsafe_trunc instead of trunc see:
    # https://github.com/CliMA/Oceananigans.jl/issues/828
    # https://github.com/CliMA/Oceananigans.jl/pull/997

    i⁻ = Base.unsafe_trunc(Int, fractional_idx)
    i⁻ = Int(i⁻ + 1) # convert to "proper" integer?
    i⁺ = i⁻ + 1
    ξ = mod(fractional_idx, 1)

    return (i⁻, i⁺, ξ)
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

@kernel function _interpolate!!(to_field, to_grid, to_location,
                                from_field, from_grid, from_location)

    i, j, k = @index(Global, NTuple)
    to_node = node(i, j, k, to_grid, to_location...)
    @inbounds to_field[i, j, k] = interpolate(to_node, from_field, from_location, from_grid)
end

"""
    interpolate!(to_field::Field, from_field::AbstractField)

Interpolate `from_field` `to_field` and then fill the halo regions of `to_field`.
"""
function interpolate!(to_field::Field, from_field::AbstractField)
    to_grid = to_field.grid

    from_arch = architecture(from_field)
    to_arch = architecture(to_field)
    if !isnothing(from_arch) && to_arch == from_arch
        msg = "Cannot interpolate! because from_field is on $from_arch while to_field is on $to_field."
        throw(ArgumentError(msg))
    end

    # Make locations
    from_location = Tuple(L() for L in location(from_field))
    to_location   = Tuple(L() for L in location(to_field))

    launch!(from_arch, to_grid, size(to_field),
            _interpolate!, to_field, to_grid, to_location,
            from_field, from_grid, from_location)

    fill_halo_regions!(to_field)

    return nothing
end

