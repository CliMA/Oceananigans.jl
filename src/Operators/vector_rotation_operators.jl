using Oceananigans.Grids: Center,
                         SphericalShellGrid,
                         λnode,
                         spherical_shell_unit_vector,
                         spherical_shell_tangent_basis,
                         φnode

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

- Cartesian coordinates for any grid that discretizes a cartesian domain (e.g. a `RectilinearGrid`)
- Geographic coordinates for any grid that discretizes a spherical domain (e.g. an `AbstractCurvilinearGrid`)

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

- Cartesian coordinates for any grid that discretizes a cartesian domain (e.g. a `RectilinearGrid`)
- Geographic coordinates for any grid that discretizes a spherical domain (e.g. an `AbstractCurvilinearGrid`)

Therefore, for the [`RectilinearGrid`](@ref) and the [`LatitudeLongitudeGrid`](@ref), the _extrinsic_ and the
_intrinsic_ coordinate systems are equivalent. However, for other grids (e.g., for the
 [`ConformalCubedSphereGrid`](@ref)) that might not be the case.
"""
@inline extrinsic_vector(i, j, k, grid::AbstractGrid, uᵢ, vᵢ, wᵢ) =
    getvalue(uᵢ, i, j, k, grid), getvalue(vᵢ, i, j, k, grid), getvalue(wᵢ, i, j, k, grid)

# 2D vectors
@inline intrinsic_vector(i, j, k, grid::AbstractGrid, uₑ, vₑ) =
    getvalue(uₑ, i, j, k, grid), getvalue(vₑ, i, j, k, grid)

@inline intrinsic_vector(i, j, k, grid::AbstractGrid, LX, LY, LZ, uₑ, vₑ) =
    intrinsic_vector(i, j, k, grid, uₑ, vₑ)

@inline extrinsic_vector(i, j, k, grid::AbstractGrid, uᵢ, vᵢ) =
    getvalue(uᵢ, i, j, k, grid), getvalue(vᵢ, i, j, k, grid)

@inline extrinsic_vector(i, j, k, grid::AbstractGrid, LX, LY, LZ, uᵢ, vᵢ) =
    extrinsic_vector(i, j, k, grid, uᵢ, vᵢ)

@inline function spherical_shell_reference_cartesian_node(i, j, k, grid::SphericalShellGrid, LX, LY, LZ)
    λ = λnode(i, j, k, grid, LX, LY, LZ)
    φ = φnode(i, j, k, grid, LX, LY, LZ)
    x̂, ŷ, ẑ = spherical_shell_unit_vector(λ, φ)
    return grid.radius * x̂, grid.radius * ŷ, grid.radius * ẑ
end

@inline function spherical_shell_covariant_basis(i, j, k, grid::SphericalShellGrid, LX, LY, LZ)
    FT = eltype(grid)
    half = convert(FT, 1//2)

    x⁺ᶦ, y⁺ᶦ, z⁺ᶦ = spherical_shell_reference_cartesian_node(i + 1, j,     k, grid, LX, LY, LZ)
    x⁻ᶦ, y⁻ᶦ, z⁻ᶦ = spherical_shell_reference_cartesian_node(i - 1, j,     k, grid, LX, LY, LZ)
    x⁺ʲ, y⁺ʲ, z⁺ʲ = spherical_shell_reference_cartesian_node(i,     j + 1, k, grid, LX, LY, LZ)
    x⁻ʲ, y⁻ʲ, z⁻ʲ = spherical_shell_reference_cartesian_node(i,     j - 1, k, grid, LX, LY, LZ)

    a₁x = half * (x⁺ᶦ - x⁻ᶦ)
    a₁y = half * (y⁺ᶦ - y⁻ᶦ)
    a₁z = half * (z⁺ᶦ - z⁻ᶦ)

    a₂x = half * (x⁺ʲ - x⁻ʲ)
    a₂y = half * (y⁺ʲ - y⁻ʲ)
    a₂z = half * (z⁺ʲ - z⁻ʲ)

    return (a₁x, a₁y, a₁z), (a₂x, a₂y, a₂z)
end

@inline spherical_shell_covariant_basis(i, j, k, grid::SphericalShellGrid) =
    spherical_shell_covariant_basis(i, j, k, grid, Center(), Center(), Center())


"""
    rotation_angle(i, j, grid::OrthogonalSphericalShellGrid)

Return the rotation angle (in radians) of the `i, j`-th point of the `grid`.
The rotation angle is the angle (positive counter-clockwise) that we need to rotate
the grid's intrinsic coordinates in order to match the grid's extrinsic coordinates.
"""
@inline function rotation_angle(i, j, grid::OrthogonalSphericalShellGrid)

    φᶠᶠᵃ⁺⁺ = φnode(i+1, j+1, 1, grid, Face(), Face(), Center())
    φᶠᶠᵃ⁺⁻ = φnode(i+1, j,   1, grid, Face(), Face(), Center())
    φᶠᶠᵃ⁻⁺ = φnode(i,   j+1, 1, grid, Face(), Face(), Center())
    φᶠᶠᵃ⁻⁻ = φnode(i,   j,   1, grid, Face(), Face(), Center())

    Δyᶠᶜᵃ⁺ = Δyᶠᶜᶜ(i+1, j,   1, grid)
    Δyᶠᶜᵃ⁻ = Δyᶠᶜᶜ(i,   j,   1, grid)
    Δxᶜᶠᵃ⁺ = Δxᶜᶠᶜ(i,   j+1, 1, grid)
    Δxᶜᶠᵃ⁻ = Δxᶜᶠᶜ(i,   j,   1, grid)

    Rcosθ₁ = ifelse(Δyᶠᶜᵃ⁺ == 0, zero(grid), deg2rad(φᶠᶠᵃ⁺⁺ - φᶠᶠᵃ⁺⁻) / Δyᶠᶜᵃ⁺)
    Rcosθ₂ = ifelse(Δyᶠᶜᵃ⁻ == 0, zero(grid), deg2rad(φᶠᶠᵃ⁻⁺ - φᶠᶠᵃ⁻⁻) / Δyᶠᶜᵃ⁻)

    # θ is the rotation angle between intrinsic and extrinsic reference frame
    Rcosθ =   (Rcosθ₁ + Rcosθ₂) / 2
    Rsinθ = - (deg2rad(φᶠᶠᵃ⁺⁺ - φᶠᶠᵃ⁻⁺) / Δxᶜᶠᵃ⁺ + deg2rad(φᶠᶠᵃ⁺⁻ - φᶠᶠᵃ⁻⁻) / Δxᶜᶠᵃ⁻) / 2

    # Normalization for the rotation angles
    R = sqrt(Rcosθ^2 + Rsinθ^2)

    cosθ, sinθ = Rcosθ / R, Rsinθ / R

    θ = atan(sinθ / cosθ)
    return θ
end

# Intrinsic and extrinsic conversion for `OrthogonalSphericalShellGrid`s,
# i.e. curvilinear grids defined on a sphere which are locally orthogonal.
# If the coordinates match with the coordinates of a latitude-longitude grid
# (i.e. globally orthogonal), these functions collapse to
# uₑ, vₑ, wₑ = uᵢ, vᵢ, wᵢ

# 2D vectors
@inline function intrinsic_vector(i, j, k, grid::OrthogonalSphericalShellGrid, uₑ, vₑ)

    u = getvalue(uₑ, i, j, k, grid)
    v = getvalue(vₑ, i, j, k, grid)

    θ = rotation_angle(i, j, grid::OrthogonalSphericalShellGrid)
    sinθ = sin(θ)
    cosθ = cos(θ)

    uᵢ = u * cosθ - v * sinθ
    vᵢ = u * sinθ + v * cosθ

    return uᵢ, vᵢ
end

# 2D vectors
@inline function intrinsic_vector(i, j, k, grid::SphericalShellGrid, LX, LY, LZ, uₑ, vₑ)
    u = getvalue(uₑ, i, j, k, grid)
    v = getvalue(vₑ, i, j, k, grid)

    eλ, eφ, _ = spherical_shell_tangent_basis(i, j, k, grid, LX, LY, LZ)
    (a₁x, a₁y, a₁z), (a₂x, a₂y, a₂z) = spherical_shell_covariant_basis(i, j, k, grid, LX, LY, LZ)

    Vx = u * eλ[1] + v * eφ[1]
    Vy = u * eλ[2] + v * eφ[2]
    Vz = u * eλ[3] + v * eφ[3]

    uᵢ = a₁x * Vx + a₁y * Vy + a₁z * Vz
    vᵢ = a₂x * Vx + a₂y * Vy + a₂z * Vz

    return uᵢ, vᵢ
end

@inline intrinsic_vector(i, j, k, grid::SphericalShellGrid, uₑ, vₑ) =
    intrinsic_vector(i, j, k, grid, Center(), Center(), Center(), uₑ, vₑ)

# 3D vectors
@inline function intrinsic_vector(i, j, k, grid::OrthogonalSphericalShellGrid, uₑ, vₑ, wₑ)

    uᵢ, vᵢ = intrinsic_vector(i, j, k, grid, uₑ, vₑ)
    wᵢ = getvalue(wₑ, i, j, k, grid)

    return uᵢ, vᵢ, wᵢ
end

# 3D vectors
@inline function intrinsic_vector(i, j, k, grid::SphericalShellGrid, uₑ, vₑ, wₑ)
    uᵢ, vᵢ = intrinsic_vector(i, j, k, grid, uₑ, vₑ)
    wᵢ = getvalue(wₑ, i, j, k, grid)

    return uᵢ, vᵢ, wᵢ
end

# 2D vectors
@inline function extrinsic_vector(i, j, k, grid::OrthogonalSphericalShellGrid, uᵢ, vᵢ)

    u = getvalue(uᵢ, i, j, k, grid)
    v = getvalue(vᵢ, i, j, k, grid)

    θ = rotation_angle(i, j, grid::OrthogonalSphericalShellGrid)
    sinθ = sin(θ)
    cosθ = cos(θ)

    uₑ = + u * cosθ + v * sinθ
    vₑ = - u * sinθ + v * cosθ

    return uₑ, vₑ
end

# 2D vectors
@inline function extrinsic_vector(i, j, k, grid::SphericalShellGrid, LX, LY, LZ, uᵢ, vᵢ)
    u₁ = getvalue(uᵢ, i, j, k, grid)
    u₂ = getvalue(vᵢ, i, j, k, grid)

    (a₁x, a₁y, a₁z), (a₂x, a₂y, a₂z) = spherical_shell_covariant_basis(i, j, k, grid, LX, LY, LZ)
    eλ, eφ, _ = spherical_shell_tangent_basis(i, j, k, grid, LX, LY, LZ)

    g₁₁ = a₁x^2 + a₁y^2 + a₁z^2
    g₁₂ = a₁x * a₂x + a₁y * a₂y + a₁z * a₂z
    g₂₂ = a₂x^2 + a₂y^2 + a₂z^2
    detg = g₁₁ * g₂₂ - g₁₂^2

    u¹ = ifelse(detg == zero(grid), zero(grid), (g₂₂ * u₁ - g₁₂ * u₂) / detg)
    u² = ifelse(detg == zero(grid), zero(grid), (g₁₁ * u₂ - g₁₂ * u₁) / detg)

    Vx = u¹ * a₁x + u² * a₂x
    Vy = u¹ * a₁y + u² * a₂y
    Vz = u¹ * a₁z + u² * a₂z

    uₑ = Vx * eλ[1] + Vy * eλ[2] + Vz * eλ[3]
    vₑ = Vx * eφ[1] + Vy * eφ[2] + Vz * eφ[3]

    return uₑ, vₑ
end

@inline extrinsic_vector(i, j, k, grid::SphericalShellGrid, uᵢ, vᵢ) =
    extrinsic_vector(i, j, k, grid, Center(), Center(), Center(), uᵢ, vᵢ)

# 3D vectors
@inline function extrinsic_vector(i, j, k, grid::OrthogonalSphericalShellGrid, uᵢ, vᵢ, wᵢ)

    uₑ, vₑ = extrinsic_vector(i, j, k, grid, uᵢ, vᵢ)
    wₑ = getvalue(wᵢ, i, j, k, grid)

    return uₑ, vₑ, wₑ
end

# 3D vectors
@inline function extrinsic_vector(i, j, k, grid::SphericalShellGrid, uᵢ, vᵢ, wᵢ)
    uₑ, vₑ = extrinsic_vector(i, j, k, grid, uᵢ, vᵢ)
    wₑ = getvalue(wᵢ, i, j, k, grid)

    return uₑ, vₑ, wₑ
end
