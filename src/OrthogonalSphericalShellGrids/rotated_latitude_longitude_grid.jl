using CubedSphere.SphericalGeometry
using Oceananigans.Grids: LatitudeLongitudeGrid, Bounded
using Oceananigans.Utils: KernelParameters
using StaticArrays
using LinearAlgebra

struct LatitudeLongitudeRotation{FT}
    north_pole :: FT
end

const RotatedLatitudeLongitudeGrid{FT, TX, TY, TZ, Z} =
    OrthogonalSphericalShellGrid{FT, TX, TY, TZ, Z,
                                 <:LatitudeLongitudeRotation} where {FT, TX, TY, TZ, Z}

# Helper function (TODO: define Operators before grids...)
@inline lat_lon_metric(m, i, j) = @inbounds m[i, j]
@inline lat_lon_metric(m::AbstractVector, i, j) = @inbounds m[j]
@inline lat_lon_metric(m::Number, j) = m
@inline lat_lon_metric(m::Number, i, j) = m

function latitude_longitude_shift((λ₀, φ₀))
    Δλ = - λ₀
    Δφ = - φ₀
    return Δλ, Δφ
end

"""
    RotatedLatitudeLongitudeGrid(arch::AbstractArchitecture = CPU(),
                                 FT::DataType = Oceananigans.defaults.FloatType;
                                 size,
                                 north_pole,
                                 longitude,
                                 latitude,
                                 z,
                                 halo = (3, 3, 3),
                                 radius = Oceananigans.defaults.planet_radius,
                                 topology = (Bounded, Bounded, Bounded))

Return a `RotatedLatitudeLongitudeGrid` with arbitrary `north_pole`, a 2-tuple
giving the longitude and latitude of the "grid north pole", which may differ from the
geographic north pole at `(0, 90)`.

Note that `longitude` and `latitude` are interpreted as applying to the grid _before_
the pole is rotated.

All other arguments are the same as for [`LatitudeLongitudeGrid`](@ref).

Example
=======

```jldoctest rllg
using Oceananigans
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid

size = (90, 40, 1)
longitude = (0, 360)
latitude = (-80, 80)
z = (0, 1)
grid = RotatedLatitudeLongitudeGrid(; size, longitude, latitude, z, north_pole=(70, 55))

# output
90×40×1 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 3×3×3 halo and with precomputed metrics
├── centered at (λ, φ) = (146.656, 11.3134)
├── longitude: Bounded  extent 360.0 degrees variably spaced with min(Δλ)=0.694593, max(Δλ)=4.0
├── latitude:  Bounded  extent 160.0 degrees variably spaced with min(Δφ)=4.0, max(Δφ)=4.0
└── z:         Bounded  z ∈ [0.0, 1.0]       regularly spaced with Δz=1.0
```

We can also make an ordinary LatitudeLongitudeGrid using `north_polar = (0, 90)`:

```jldoctest rllg
grid = RotatedLatitudeLongitudeGrid(; size, longitude, latitude, z, north_pole=(0, 90))

# output
90×40×1 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 3×3×3 halo and with precomputed metrics
├── centered at (λ, φ) = (180.0, 0.0)
├── longitude: Bounded  extent 360.0 degrees variably spaced with min(Δλ)=0.694593, max(Δλ)=4.0
├── latitude:  Bounded  extent 160.0 degrees variably spaced with min(Δφ)=4.0, max(Δφ)=4.0
└── z:         Bounded  z ∈ [0.0, 1.0]       regularly spaced with Δz=1.0
```
"""
function RotatedLatitudeLongitudeGrid(arch::AbstractArchitecture = CPU(),
                                      FT::DataType = Oceananigans.defaults.FloatType;
                                      size,
                                      north_pole,
                                      longitude,
                                      latitude,
                                      z,
                                      halo = (3, 3, 3),
                                      radius = Oceananigans.defaults.planet_radius,
                                      topology = (Bounded, Bounded, Bounded))

    _, φ₀ = north_pole

    if φ₀ < 0
        throw(ArgumentError("North pole latitude must be >= 0."))
    elseif φ₀ > 90
        throw(ArgumentError("North pole latitude must be <= 90 degrees."))
    end

    shifted_halo = halo .+ 1
    source_grid = LatitudeLongitudeGrid(arch, FT; size, z, topology, radius,
                                        latitude, longitude, halo = shifted_halo)

    conformal_mapping = LatitudeLongitudeRotation(north_pole)
    grid = OrthogonalSphericalShellGrid(arch, FT; size, z, radius, halo, topology, conformal_mapping)
    rotate_metrics!(grid, source_grid)

    return grid
end

function rotate_metrics!(grid, shifted_lat_lon_grid)
    arch = architecture(grid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    parameters = KernelParameters(-Hx:Nx+Hx+1, -Hy:Ny+Hy+1)
    launch!(arch, grid, parameters, _rotate_metrics!, grid, shifted_lat_lon_grid)
    return nothing
end

# Convert from Spherical to Cartesian
function spherical_to_cartesian(φ, λ; radius = 1, check_latitude_bounds = true)
    check_latitude_bounds && abs(φ) > π/2 && error("Latitude φ must be within -90 ≤ φ ≤ 90 degrees.")
    x = radius * cos(λ) * cos(φ)
    y = radius * sin(λ) * cos(φ)
    z = radius * sin(φ)
    return x, y, z
end

# Convert from Cartesian to Spherical
function cartesian_to_spherical(x, y, z)
    φ = atan(z, sqrt(x*x + y*y))
    λ = atan(y, x)
    return φ, λ
end

function cartesian_to_spherical(X)
    x, y, z = X
    return cartesian_to_spherical(x, y, z)
end

# Rotation about x-axis by dλ (Change in Longitude)
x_rotation_matrix(dλ) = @SMatrix [1        0       0
                                  0  cos(dλ) -sin(dλ)
                                  0  sin(dλ)  cos(dλ)]

# Rotation about y-axis by dφ (Change in Latitude)
y_rotation_matrix(dφ) = @SMatrix [ cos(dφ)  0  sin(dφ)
                                   0        1  0
                                  -sin(dφ)  0  cos(dφ)]

# Rotation about z-axis by dλ (Change in Longitude)
z_rotation_matrix(dλ) = @SMatrix [cos(dλ) -sin(dλ) 0
                                  sin(dλ)  cos(dλ) 0
                                  0        0       1]

"""
    rotate_coordinates(λ′, φ′, λ₀, φ₀)

Return the geographic longitude and latitude `(λ, φ)` corresponding to the rotated
coordinates `(λ′, φ′)` on a grid whose north pole is located at `(λ₀, φ₀)`. All
arguments are interpreted in degrees. The rotation aligns the grid pole with the
geographic pole, then maps the rotated point back to geographic coordinates.
"""
function rotate_coordinates(λ′, φ′, λ₀, φ₀)
    λ′ *= π / 180
    φ′ *= π / 180
    λ₀ *= π / 180
    φ₀ *= π / 180

    dλ = λ₀
    dφ = π/2 - φ₀

    # Convert to Cartesian
    X′ = SVector(spherical_to_cartesian(φ′, λ′; check_latitude_bounds = false)...)

    # Rotate Cartesian coordinates
    Ry = y_rotation_matrix(dφ)
    Rz = z_rotation_matrix(dλ)
    X = Rz * Ry * X′

    # Convert back to Spherical
    φ, λ = cartesian_to_spherical(X)

    λ *= 180 / π
    φ *= 180 / π

    return λ, φ
end

@kernel function _rotate_metrics!(grid, source_grid)
    i, j = @index(Global, NTuple)

    λ₀, φ₀ = grid.conformal_mapping.north_pole

    @inbounds begin
        # Shifted metrics
        λ′ = source_grid.λᶜᵃᵃ[i]
        φ′ = lat_lon_metric(source_grid.φᵃᶜᵃ, i, j)
        λ, φ = rotate_coordinates(λ′, φ′, λ₀, φ₀)
        grid.λᶜᶜᵃ[i, j]  = λ
        grid.φᶜᶜᵃ[i, j]  = φ

        λ′ = source_grid.λᶠᵃᵃ[i]
        φ′ = lat_lon_metric(source_grid.φᵃᶜᵃ, i, j)
        λ, φ = rotate_coordinates(λ′, φ′, λ₀, φ₀)
        grid.λᶠᶜᵃ[i, j] = λ
        grid.φᶠᶜᵃ[i, j] = φ

        λ′ = source_grid.λᶜᵃᵃ[i]
        φ′ = lat_lon_metric(source_grid.φᵃᶠᵃ, i, j)
        λ, φ = rotate_coordinates(λ′, φ′, λ₀, φ₀)
        grid.λᶜᶠᵃ[i, j]  = λ
        grid.φᶜᶠᵃ[i, j]  = φ

        λ′ = source_grid.λᶠᵃᵃ[i]
        φ′ = lat_lon_metric(source_grid.φᵃᶠᵃ, i, j)
        λ, φ = rotate_coordinates(λ′, φ′, λ₀, φ₀)
        grid.λᶠᶠᵃ[i, j] = λ
        grid.φᶠᶠᵃ[i, j] = φ

        # Directly copiable metrics:
        grid.Δxᶜᶜᵃ[i, j] = lat_lon_metric(source_grid.Δxᶜᶜᵃ, i, j)
        grid.Δxᶠᶜᵃ[i, j] = lat_lon_metric(source_grid.Δxᶠᶜᵃ, i, j)
        grid.Δxᶜᶠᵃ[i, j] = lat_lon_metric(source_grid.Δxᶜᶠᵃ, i, j)
        grid.Δxᶠᶠᵃ[i, j] = lat_lon_metric(source_grid.Δxᶠᶠᵃ, i, j)

        grid.Azᶜᶜᵃ[i, j] = lat_lon_metric(source_grid.Azᶜᶜᵃ, i, j)
        grid.Azᶠᶜᵃ[i, j] = lat_lon_metric(source_grid.Azᶠᶜᵃ, i, j)
        grid.Azᶜᶠᵃ[i, j] = lat_lon_metric(source_grid.Azᶜᶠᵃ, i, j)
        grid.Azᶠᶠᵃ[i, j] = lat_lon_metric(source_grid.Azᶠᶠᵃ, i, j)

        grid.Δyᶜᶠᵃ[i, j] = lat_lon_metric(source_grid.Δyᶜᶠᵃ, i, j)
        grid.Δyᶠᶜᵃ[i, j] = lat_lon_metric(source_grid.Δyᶠᶜᵃ, i, j)

        # Note transposition of location
        grid.Δyᶜᶜᵃ[i, j] = lat_lon_metric(source_grid.Δyᶠᶜᵃ, i, j)
        grid.Δyᶠᶠᵃ[i, j] = lat_lon_metric(source_grid.Δyᶜᶠᵃ, i, j)
    end
end
