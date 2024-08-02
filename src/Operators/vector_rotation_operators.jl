# TODO: have a general Oceananigans-wide function that retrieves a pointwise
# value for a function, an array, a number, a field etc?
# This would be a generalization of `getbc` that could be used everywhere we need it
@inline getvalue(::Nothing,        i, j, k, grid, args...) = nothing
@inline getvalue(a::Number,        i, j, k, grid, args...) = a
@inline getvalue(a::AbstractArray, i, j, k, grid, args...) = @inbounds a[i, j, k]

"""
    intrinsic_vector(i, j, k, grid::AbstractGrid, uₑ, vₑ, wₑ)

Convert the three-dimensional vector with components `uₑ, vₑ, wₑ` defined in an _extrinsic_ 
coordinate system associated with the domain, to the coordinate system _intrinsic_ to the grid. 

_extrinsic_ coordinate systems are:

- Cartesian for any grid that discretizes a Cartesian domain (e.g. a `RectilinearGrid`)
- Geographic coordinates for any grid that discretizes a Spherical domain (e.g. an `AbstractCurvilinearGrid`)

Therefore, for the [`RectilinearGrid`](@ref) and the [`LatitudeLongitudeGrid`](@ref), the _extrinsic_ and the 
_intrinsic_ coordinate system are equivalent. However, for other grids (e.g., for the
 [`ConformalCubedSphereGrid`](@ref)) that might not be the case.
"""
@inline intrinsic_vector(i, j, k, grid::AbstractGrid, uₑ, vₑ, wₑ) = 
    getvalue(uₑ, i, j, k, grid), getvalue(vₑ, i, j, k, grid), getvalue(wₑ, i, j, k, grid)

"""
    extrinsic_vector(i, j, k, grid::AbstractGrid, uᵢ, vᵢ, wᵢ)

Convert the three-dimensional vector with components `uᵢ, vᵢ, wᵢ ` defined on the _intrinsic_ coordinate
system of the grid, to the _extrinsic_ coordinate system associated with the domain.

_extrinsic_ coordinate systems are:

- Cartesian for any grid that discretizes a Cartesian domain (e.g. a `RectilinearGrid`)
- Geographic coordinates for any grid that discretizes a Spherical domain (e.g. an `AbstractCurvilinearGrid`)

Therefore, for the [`RectilinearGrid`](@ref) and the [`LatitudeLongitudeGrid`](@ref), the _extrinsic_ and the 
_intrinsic_ coordinate systems are equivalent. However, for other grids (e.g., for the
 [`ConformalCubedSphereGrid`](@ref)) that might not be the case.
"""
@inline extrinsic_vector(i, j, k, grid::AbstractGrid, uᵢ, vᵢ, wᵢ) =
    getvalue(uᵢ, i, j, k, grid), getvalue(vᵢ, i, j, k, grid), getvalue(wᵢ, i, j, k, grid)

# 2D vectors
@inline intrinsic_vector(i, j, k, grid::AbstractGrid, uₑ, vₑ) = 
    getvalue(uₑ, i, j, k, grid), getvalue(vₑ, i, j, k, grid)

@inline extrinsic_vector(i, j, k, grid::AbstractGrid, uᵢ, vᵢ) = 
    getvalue(uᵢ, i, j, k, grid), getvalue(vᵢ, i, j, k, grid)

# Intrinsic and extrinsic conversion for `OrthogonalSphericalShellGrid`s,
# i.e. curvilinear grids defined on a sphere which are locally orthogonal.
# If the coordinates match with the coordinates of a latitude-longitude grid
# (i.e. globally orthogonal), these functions collapse to 
# uₑ, vₑ, wₑ = uᵢ, vᵢ, wᵢ

# 2D vectors
@inline function intrinsic_vector(i, j, k, grid::OrthogonalSphericalShellGrid, uₑ, vₑ)

    φᶜᶠᵃ₊ = φnode(i, j+1, 1, grid, Center(), Face(), Center())
    φᶜᶠᵃ₋ = φnode(i,   j, 1, grid, Center(), Face(), Center())
    Δyᶜᶜᵃ = Δyᶜᶜᶜ(i,   j, 1, grid)

    # θᵢ is the rotation angle between intrinsic and extrinsic reference frame
    Rcosθᵢ = deg2rad(φᶜᶠᵃ₊ - φᶜᶠᵃ₋) / Δyᶜᶜᵃ

    φᶠᶜᵃ₊ = φnode(i+1, j, 1, grid, Face(), Center(), Center())
    φᶠᶜᵃ₋ = φnode(i,   j, 1, grid, Face(), Center(), Center())
    Δxᶜᶜᵃ = Δxᶜᶜᶜ(i,   j, 1, grid)

    Rsinθᵢ = - deg2rad(φᶠᶜᵃ₊ - φᶠᶜᵃ₋) / Δxᶜᶜᵃ

    # Normalization for the rotation angles
    Rᵢ = sqrt(Rcosθᵢ^2 + Rsinθᵢ^2)

    u  = getvalue(uₑ, i, j, k, grid)
    v  = getvalue(vₑ, i, j, k, grid)

    cosθᵢ = Rcosθᵢ / Rᵢ
    sinθᵢ = Rsinθᵢ / Rᵢ

    uᵢ = u * cosθᵢ + v * sinθᵢ
    vᵢ = u * sinθᵢ - v * cosθᵢ

    return uᵢ, vᵢ
end

# 3D vectors
@inline function intrinsic_vector(i, j, k, grid::OrthogonalSphericalShellGrid, uₑ, vₑ, wₑ)

    uᵢ, vᵢ = intrinsic_vector(i, j, k, grid, uₑ, vₑ)
    wᵢ = getvalue(wₑ, i, j, k, grid)

    return uᵢ, vᵢ, wᵢ
end

# 2D vectors
@inline function extrinsic_vector(i, j, k, grid::OrthogonalSphericalShellGrid, uᵢ, vᵢ)

    φᶜᶠᵃ₊ = φnode(i, j+1, 1, grid, Center(), Face(), Center())
    φᶜᶠᵃ₋ = φnode(i,   j, 1, grid, Center(), Face(), Center())
    Δyᶜᶜᵃ = Δyᶜᶜᶜ(i,   j, 1, grid)

    # θₑ is the rotation angle between intrinsic and extrinsic reference frame
    Rcosθₑ = deg2rad(φᶜᶠᵃ₊ - φᶜᶠᵃ₋) / Δyᶜᶜᵃ

    φᶠᶜᵃ₊ = φnode(i+1, j, 1, grid, Face(), Center(), Center())
    φᶠᶜᵃ₋ = φnode(i,   j, 1, grid, Face(), Center(), Center())
    Δxᶜᶜᵃ = Δxᶜᶜᶜ(i,   j, 1, grid)

    Rsinθₑ = - deg2rad(φᶠᶜᵃ₊ - φᶠᶜᵃ₋) / Δxᶜᶜᵃ

    # Normalization for the rotation angles
    Rₑ = sqrt(Rcosθₑ^2 + Rsinθₑ^2)

    u  = getvalue(uᵢ, i, j, k, grid)
    v  = getvalue(vᵢ, i, j, k, grid)

    cosθₑ = Rcosθₑ / Rₑ
    sinθₑ = Rsinθₑ / Rₑ

    uₑ = u * cosθₑ - v * sinθₑ
    vₑ = u * sinθₑ + v * cosθₑ

    return uₑ, vₑ
end

# 3D vectors
@inline function extrinsic_vector(i, j, k, grid::OrthogonalSphericalShellGrid, uᵢ, vᵢ, wᵢ)

    uₑ, vₑ = intrinsic_vector(i, j, k, grid, uᵢ, vᵢ)
    wₑ = getvalue(wᵢ, i, j, k, grid)

    return uₑ, vₑ, wₑ
end