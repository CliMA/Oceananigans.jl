using Oceananigans.Grids: topology, node, _node, φnode, λnode,
                          xspacings, yspacings, zspacings, λspacings, φspacings,
                          XFlatGrid, YFlatGrid, ZFlatGrid,
                          XYFlatGrid, YZFlatGrid, XZFlatGrid,
                          XRegularRG, YRegularRG, ZRegularRG,
                          XRegularLLG, YRegularLLG, ZRegularLLG,
                          ZRegOrthogonalSphericalShellGrid,
                          RectilinearGrid, LatitudeLongitudeGrid

using Oceananigans.Operators: Δx, Δy, Δz

using Oceananigans.Architectures: child_architecture

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
@inline function index_binary_search(vec, val::Number, N)
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
    i₁, i₂ = index_binary_search(vec, val, N)

    @inbounds x₁ = vec[i₁]
    @inbounds x₂ = vec[i₂]

    ii = (i₂ - i₁) / (x₂ - x₁) * (val - x₁) + i₁
    ii = ifelse(i₁ == i₂, i₁, ii)

    FT = typeof(val) # convert "fractional index" to type of `val`
    return convert(FT, ii)
end

#####
##### Note: interpolation on LatitudeLongitudeGrid assumes a thin shell
##### (i.e. curvature effects are not incorporated when interpolating).
##### Use other methods if a more accurate interpolation is required.
#####

@inline fractional_x_index(x, locs, grid::XFlatGrid) = zero(grid)

@inline function fractional_x_index(x, locs, grid::XRegularRG)
    x₀ = xnode(1, 1, 1, grid, locs...)
    dx = Δx(1, 1, 1, grid, locs...)
    FT = eltype(grid)
    return convert(FT, (x - x₀) / dx) + 1 # 1 - based indexing
end

@inline function fractional_x_index(x, locs, grid::RectilinearGrid)
    loc = @inbounds locs[1]
     Tx = topology(grid, 1)()
     Nx = length(loc, Tx, grid.Nx)
     xn = xnodes(grid, locs...)
     FT = eltype(grid)
    return convert(FT, fractional_index(x, xn, Nx))
end

# Because of precision errors with numbers close to 0,
# we need to make sure we approach the correct limit also from the left.
@inline function convert_to_0_360(x::FT) where FT
    x₀ = ((x % 360) + 360) % 360
    x⁻ = - eps(convert(FT, 360))
    return ifelse(x⁻ ≤ x < 0, 360 + x, x₀)
end

# No need to check precision for integers
@inline convert_to_0_360(x::Integer) = ((x % 360) + 360) % 360

# Find n for which 360 * n ≤ λ ≤ 360 * (n + 1)
@inline find_λ_range(λ) = ifelse((λ < 0) & (mod(λ, 360) != 0), λ ÷ 360 - 1, λ ÷ 360)

# Convert x to lie in the λ₀ : λ₀ + 360 range by accounting for the cyclic
# nature of the longitude coordinate.
@inline function convert_to_λ₀_λ₀_plus360(x, λ₀)
    x  = convert_to_0_360(x)
    n  = find_λ_range(λ₀)
    λ⁻ = convert_to_0_360(λ₀)
    n  = ifelse(x ≥ λ⁻, n, n+1)
    return x + 360 * n
end

# When interpolating longitude values, we convert the longitude to
# interpolate to lie in the λ₀ : λ₀ + 360 range, where λ₀ is the westernmost node
# of the interpolating grid.
@inline function fractional_x_index(λ, locs, grid::XRegularLLG)
    λ₀ = λnode(1, 1, 1, grid, locs...)
    λ₁ = λnode(2, 1, 1, grid, locs...)
    Δλ = λ₁ - λ₀
    λc = convert_to_λ₀_λ₀_plus360(λ, λ₀ - Δλ/2) # Making sure we have the right range
    FT = eltype(grid)
    return convert(FT, (λc - λ₀) / (λ₁ - λ₀)) + 1 # 1 - based indexing
end

# When interpolating longitude values, we convert the longitude to
# interpolate to lie in the λ₀ : λ₀ + 360 range, where λ₀ is the westernmost node
# of the interpolating grid.
@inline function fractional_x_index(λ, locs, grid::LatitudeLongitudeGrid)
    loc = @inbounds locs[1]
     Tλ = topology(grid, 1)()
     Nλ = length(loc, Tλ, grid.Nx)
     λn = λnodes(grid, locs...)
     λ₀ = @inbounds λn[1]
     λ₁ = @inbounds λn[2]
     Δλ = λ₁ - λ₀
     λc = convert_to_λ₀_λ₀_plus360(λ, λ₀ - Δλ/2)
     FT = eltype(grid)
    return convert(FT, fractional_index(λc, λn, Nλ))
end

@inline fractional_y_index(y, locs, grid::YFlatGrid) = zero(grid)

@inline function fractional_y_index(y, locs, grid::YRegularRG)
    y₀ = ynode(1, 1, 1, grid, locs...)
    dy = Δy(1, 1, 1, grid, locs...)
    FT = eltype(grid)
    return convert(FT, (y - y₀) / dy) + 1 # 1 - based indexing
end

@inline function fractional_y_index(φ, locs, grid::YRegularLLG)
    φ₀ = φnode(1, 1, 1, grid, locs...)
    φ₁ = φnode(1, 2, 1, grid, locs...)
    FT = eltype(grid)
    return convert(FT, (φ - φ₀) / (φ₁ - φ₀)) + 1 # 1 - based indexing
end

@inline function fractional_y_index(y, locs, grid::RectilinearGrid)
    loc = @inbounds locs[2]
     Ty = topology(grid, 2)()
     Ny = length(loc, Ty, grid.Ny)
     yn = ynodes(grid, locs...)
     FT = eltype(grid)
    return convert(FT, fractional_index(y, yn, Ny))
end

@inline function fractional_y_index(y, locs, grid::LatitudeLongitudeGrid)
    loc = @inbounds locs[2]
     Ty = topology(grid, 2)()
     Ny = length(loc, Ty, grid.Ny)
     yn = φnodes(grid, locs...)
     FT = eltype(grid)
    return convert(FT, fractional_index(y, yn, Ny))
end

@inline fractional_z_index(z, locs, grid::ZFlatGrid) = zero(grid)

ZRegGrid = Union{ZRegularRG, ZRegularLLG, ZRegOrthogonalSphericalShellGrid}

@inline function fractional_z_index(z::FT, locs, grid::ZRegGrid) where FT
    z₀ = znode(1, 1, 1, grid, locs...)
    dz = Δz(1, 1, 1, grid, locs...)
    return convert(FT, (z - z₀) / dz) + 1 # 1 - based indexing
end

@inline function fractional_z_index(z, locs, grid)
    loc = @inbounds locs[3]
     Tz = topology(grid, 3)()
     Nz = length(loc, Tz, grid.Nz)
     zn = znodes(grid, loc)
     FT = eltype(grid)
    return convert(FT, fractional_index(z, zn, Nz))
end

struct FractionalIndices{I, J, K}
    i :: I
    j :: J
    k :: K
end

"""
    FractionalIndices(x, y, z, grid, loc...)

Convert the coordinates `(x, y, z)` to _fractional_ indices on a regular rectilinear grid
located at `loc`, where `loc` is a 3-tuple of `Center` and `Face`. Fractional indices are
floats indicating a location between grid points.
"""
@inline FractionalIndices(at_node, grid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, ℓx, ℓy, ℓz)

@inline FractionalIndices(at_node, grid::XFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, nothing, ℓy, ℓz)
@inline FractionalIndices(at_node, grid::YFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, ℓx, nothing, ℓz)
@inline FractionalIndices(at_node, grid::ZFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, ℓx, ℓy, nothing)

@inline FractionalIndices(at_node, grid::XYFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, nothing, nothing, ℓz)
@inline FractionalIndices(at_node, grid::YZFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, ℓx, nothing, nothing)
@inline FractionalIndices(at_node, grid::XZFlatGrid, ℓx, ℓy, ℓz) = _fractional_indices(at_node, grid, nothing, ℓy, nothing)

@inline function _fractional_indices((x, y, z), grid, ℓx, ℓy, ℓz)
    ii = fractional_x_index(x, (ℓx, ℓy, ℓz), grid)
    jj = fractional_y_index(y, (ℓx, ℓy, ℓz), grid)
    kk = fractional_z_index(z, (ℓx, ℓy, ℓz), grid)
    return FractionalIndices(ii, jj, kk)
end

@inline function _fractional_indices((y, z), grid, ::Nothing, ℓy, ℓz)
    jj = fractional_y_index(y, (nothing, ℓy, ℓz), grid)
    kk = fractional_z_index(z, (nothing, ℓy, ℓz), grid)
    return FractionalIndices(nothing, jj, kk)
end

@inline function _fractional_indices((x, z), grid, ℓx, ::Nothing, ℓz)
    ii = fractional_x_index(x, (ℓx, nothing, ℓz), grid)
    kk = fractional_z_index(z, (ℓx, nothing, ℓz), grid)
    return FractionalIndices(ii, nothing, kk)
end

@inline function _fractional_indices((x, y), grid, ℓx, ℓy, ::Nothing)
    ii = fractional_x_index(x, (ℓx, ℓy, nothing), grid)
    jj = fractional_y_index(y, (ℓx, ℓy, nothing), grid)
    return FractionalIndices(ii, jj, nothing)
end

@inline function _fractional_indices((x,), grid, ℓx, ::Nothing, ::Nothing)
    loc = (ℓx, nothing, nothing)
    ii = fractional_x_index(x, loc, grid)
    jj = nothing
    kk = nothing
    return FractionalIndices(ii, jj, kk)
end

@inline function _fractional_indices((y,), grid, ::Nothing, ℓy, ::Nothing)
    loc = (nothing, ℓy, nothing)
    ii = nothing
    jj = fractional_y_index(y, loc, grid)
    kk = nothing
    return FractionalIndices(ii, jj, kk)
end

@inline function _fractional_indices((z,), grid, ::Nothing, ::Nothing, ℓz)
    loc = (nothing, nothing, ℓz)
    ii = nothing
    jj = nothing
    kk = fractional_z_index(z, loc, grid)
    return FractionalIndices(ii, jj, kk)
end

@inline _fractional_indices(at_node, grid, ::Nothing, ::Nothing, ::Nothing) = FractionalIndices(nothing, nothing, nothing)

"""
    interpolate(at_node, from_field, from_loc, from_grid)

Interpolate `from_field`, `at_node`, on `from_grid` and at `from_loc`ation,
where `at_node` is a tuple of coordinates and and `from_loc = (ℓx, ℓy, ℓz)`.

Note that this is a lower-level `interpolate` method defined for use in CPU/GPU kernels.
"""
@inline function interpolate(at_node, from_field, from_loc, from_grid)
    fractional_indices = FractionalIndices(at_node, from_grid, from_loc...)
    return interpolate(fractional_indices, from_field, from_loc, from_grid)
end

@inline function interpolate(fidx::FractionalIndices, from_field, from_loc, from_grid)
    ix = interpolator(fidx.i)
    iy = interpolator(fidx.j)
    iz = interpolator(fidx.k)
    return _interpolate(from_field, ix, iy, iz)
end

"""
    interpolator(fractional_idx)

Return an ``interpolator tuple'' from the fractional index `fractional_idx`
defined as the 3-tuple

```
(i⁻, i⁺, ξ)
```

where `i⁻` is the index to the left of `i`, `i⁺` is the index to the
right of `i`, and `ξ` is the fractional distance between `i` and the
left bound `i⁻`, such that `ξ ∈ [0, 1)`.
"""
@inline function interpolator(fractional_idx)
    # For why we use Base.unsafe_trunc instead of trunc see:
    # https://github.com/CliMA/Oceananigans.jl/issues/828
    # https://github.com/CliMA/Oceananigans.jl/pull/997

    i⁻ = Base.unsafe_trunc(Int, fractional_idx)
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

@inline function _interpolate(data, ix, iy, iz, in...)
    # Unpack the "interpolators"
    i⁻, i⁺, ξ = ix
    j⁻, j⁺, η = iy
    k⁻, k⁺, ζ = iz

    return @inbounds ϕ₁(ξ, η, ζ) * getindex(data, i⁻, j⁻, k⁻, in...) +
                     ϕ₂(ξ, η, ζ) * getindex(data, i⁻, j⁻, k⁺, in...) +
                     ϕ₃(ξ, η, ζ) * getindex(data, i⁻, j⁺, k⁻, in...) +
                     ϕ₄(ξ, η, ζ) * getindex(data, i⁻, j⁺, k⁺, in...) +
                     ϕ₅(ξ, η, ζ) * getindex(data, i⁺, j⁻, k⁻, in...) +
                     ϕ₆(ξ, η, ζ) * getindex(data, i⁺, j⁻, k⁺, in...) +
                     ϕ₇(ξ, η, ζ) * getindex(data, i⁺, j⁺, k⁻, in...) +
                     ϕ₈(ξ, η, ζ) * getindex(data, i⁺, j⁺, k⁺, in...)
end

"""
    interpolate(to_node, from_field)

Interpolate `field` to the physical point `(x, y, z)` using trilinear interpolation.
"""
@inline function interpolate(to_node, from_field)
    from_loc = Tuple(L() for L in location(from_field))
    return interpolate(to_node, from_field, from_loc, from_field.grid)
end

@inline flatten_node(x, y, z) = (x, y, z)

@inline flatten_node(::Nothing, y, z) = flatten_node(y, z)
@inline flatten_node(x, ::Nothing, z) = flatten_node(x, z)
@inline flatten_node(x, y, ::Nothing) = flatten_node(x, y)

@inline flatten_node(x, ::Nothing, ::Nothing) = tuple(x)
@inline flatten_node(::Nothing, y, ::Nothing) = tuple(y)
@inline flatten_node(::Nothing, ::Nothing, z) = tuple(z)

@inline flatten_node(::Nothing, ::Nothing, ::Nothing) = tuple()

@inline flatten_node(x, y) = (x, y)
@inline flatten_node(::Nothing, y) = flatten_node(y)
@inline flatten_node(x, ::Nothing) = flatten_node(x)

@inline flatten_node(x) = tuple(x)
@inline flatten_node(::Nothing) = tuple()

@kernel function _interpolate!(to_field, to_grid, to_location,
                               from_field, from_grid, from_location)

    i, j, k = @index(Global, NTuple)

    to_node = _node(i, j, k, to_grid, to_location...)
    to_node = flatten_node(to_node...)

    @inbounds to_field[i, j, k] = interpolate(to_node, from_field, from_location, from_grid)
end

"""
    interpolate!(to_field::Field, from_field::AbstractField)

Interpolate `from_field` `to_field` and then fill the halo regions of `to_field`.
"""
function interpolate!(to_field::Field, from_field::AbstractField)
    to_grid   = to_field.grid
    from_grid = from_field.grid

    to_arch   = architecture(to_field)
    from_arch = architecture(from_field)

    # In case architectures are `Distributed` we
    # verify that the fields are on the same child architecture
    to_arch   = child_architecture(to_arch)
    from_arch = child_architecture(from_arch)

    if !isnothing(from_arch) && to_arch != from_arch
        msg = "Cannot interpolate! because from_field is on $from_arch while to_field is on $to_arch."
        throw(ArgumentError(msg))
    end

    # Make locations
    from_location = Tuple(L() for L in location(from_field))
    to_location   = Tuple(L() for L in location(to_field))

    params = KernelParameters(interior_indices(to_field))

    launch!(to_arch, to_grid, params,
            _interpolate!, to_field, to_grid, to_location,
            from_field, from_grid, from_location)

    fill_halo_regions!(to_field)

    return to_field
end

