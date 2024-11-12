using Distances

device = Oceananigans.Architectures.device

@inline convert_to_0_360(x) = ((x % 360) + 360) % 360

""" a structure to represent a tripolar grid on a spherical shell """
struct Tripolar{N, F, S}
    north_poles_latitude :: N
    first_pole_longitude :: F
    southernmost_latitude :: S
end

Adapt.adapt_structure(to, t::Tripolar) = 
    Tripolar(Adapt.adapt(to, t.north_poles_latitude),
             Adapt.adapt(to, t.first_pole_longitude),
             Adapt.adapt(to, t.southernmost_latitude))

const TripolarGrid{FT, TX, TY, TZ, A, R, FR, Arch} = OrthogonalSphericalShellGrid{FT, TX, TY, TZ, A, R, FR, <:Tripolar, Arch}

"""
    TripolarGrid(arch = CPU(), FT::DataType = Float64;
                 size,
                 southernmost_latitude = -80,
                 halo = (4, 4, 4),
                 radius = R_Earth,
                 z = (0, 1),
                 north_poles_latitude = 55,
                 first_pole_longitude = 70)

Construct a tripolar grid on a spherical shell.

!!! warning "Longitude coordinate must have even number of cells"
    `size` is a 3-tuple of the grid size in longitude, latitude, and vertical directions.
    Due to requirements of the folding at the north edge of the domain, the longitude size
    of the grid (i.e., the first component of `size`) _must_ be an even number!

Positional Arguments
====================

- `arch`: The architecture to use for the grid. Default is `CPU()`.
- `FT::DataType`: The data type to use for the grid. Default is `Float64`.

Keyword Arguments
=================

- `size`: The number of cells in the (longitude, latitude, vertical) dimensions.
- `southernmost_latitude`: The southernmost `Center` latitude of the grid. Default is -80.
- `halo`: The halo size in the (longitude, latitude, vertical) dimensions. Default is (4, 4, 4).
- `radius`: The radius of the spherical shell. Default is `R_Earth`.
- `z`: The vertical ``z``-coordinate range of the grid. Default is (0, 1).
- `first_pole_longitude`: The longitude of the first "north" singularity.
                          The second singularity is located at `first_pole_longitude + 180ᵒ`.
- `north_poles_latitude`: The latitude of the "north" singularities.

Return
======

An `OrthogonalSphericalShellGrid` object representing a tripolar grid on the sphere. 
The north singularities are located at

`i = 1, j = Nφ` and `i = Nλ ÷ 2 + 1, j = Nλ` 
"""
function TripolarGrid(arch = CPU(), FT::DataType = Float64; 
                      size,
                      southernmost_latitude = -80, # The southermost `Center` latitude of the grid
                      halo = (4, 4, 4),
                      radius = R_Earth,
                      z = (0, 1),
                      north_poles_latitude = 55,
                      first_pole_longitude = 70)  # The second pole is at `λ = first_pole_longitude + 180ᵒ`

    # TODO: change a couple of allocations here and there to be able
    # to construct the grid on the GPU. This is not a huge problem as
    # grid generation is quite fast, but it might become for sub-kilometer grids

    latitude  = (southernmost_latitude, 90)
    longitude = (-180, 180) 

    focal_distance = tand((90 - north_poles_latitude) / 2)

    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo

    if isodd(Nλ)
        throw(ArgumentError("The number of cells in the longitude dimension should be even!"))
    end

    # the λ and Z coordinate is the same as for the other grids,
    # but for the φ coordinate we need to remove one point at the north
    # because the the north pole is a `Center`point, not on `Face` point...
    Lx, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, Periodic(), Nλ, Hλ, longitude, :longitude, CPU())
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT,  Bounded(), Nz, Hz, z,         :z,         CPU())

    # The φ coordinate is a bit more complicated because the center points start from
    # southernmost_latitude and end at 90ᵒ N.
    φᵃᶜᵃ = collect(range(southernmost_latitude, 90, length = Nφ))
    Δφ = φᵃᶜᵃ[2] - φᵃᶜᵃ[1]
    φᵃᶠᵃ = φᵃᶜᵃ .- Δφ / 2

    # Start with the NH stereographic projection
    # TODO: make these on_architecture(arch, zeros(Nx, Ny))
    # to build the grid on GPU
    λFF = zeros(Nλ, Nφ)
    φFF = zeros(Nλ, Nφ)
    λFC = zeros(Nλ, Nφ)
    φFC = zeros(Nλ, Nφ)

    λCF = zeros(Nλ, Nφ)
    φCF = zeros(Nλ, Nφ)
    λCC = zeros(Nλ, Nφ)
    φCC = zeros(Nλ, Nφ)

    loop! = _compute_tripolar_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))

    loop!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC, 
          λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶠᵃ, φᵃᶜᵃ, 
          first_pole_longitude,
          focal_distance, Nλ)

    # We need to circshift eveything to have the first pole at the beginning of the 
    # grid and the second pole in the middle
    shift = Nλ÷4

    λFF = circshift(λFF, (shift, 0))
    φFF = circshift(φFF, (shift, 0)) 
    λFC = circshift(λFC, (shift, 0)) 
    φFC = circshift(φFC, (shift, 0)) 
    λCF = circshift(λCF, (shift, 0)) 
    φCF = circshift(φCF, (shift, 0)) 
    λCC = circshift(λCC, (shift, 0)) 
    φCC = circshift(φCC, (shift, 0))

    Nx = Nλ
    Ny = Nφ
            
    # Allocate Metrics
    # TODO: make these on_architecture(arch, zeros(Nx, Ny))
    # to build the grid on GPU
    Δxᶜᶜᵃ = zeros(Nx, Ny)
    Δxᶠᶜᵃ = zeros(Nx, Ny)
    Δxᶜᶠᵃ = zeros(Nx, Ny)
    Δxᶠᶠᵃ = zeros(Nx, Ny)

    Δyᶜᶜᵃ = zeros(Nx, Ny)
    Δyᶠᶜᵃ = zeros(Nx, Ny)
    Δyᶜᶠᵃ = zeros(Nx, Ny)
    Δyᶠᶠᵃ = zeros(Nx, Ny)

    Azᶜᶜᵃ = zeros(Nx, Ny)
    Azᶠᶜᵃ = zeros(Nx, Ny)
    Azᶜᶠᵃ = zeros(Nx, Ny)
    Azᶠᶠᵃ = zeros(Nx, Ny)

    # Calculate metrics
    loop! = _calculate_metrics!(device(CPU()), (16, 16), (Nx, Ny))

    loop!(Δxᶠᶜᵃ, Δxᶜᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
          Δyᶠᶜᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
          Azᶠᶜᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
          λᶠᶜᵃ, λᶜᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
          φᶠᶜᵃ, φᶜᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ,
          radius)
          
    # Metrics fields to fill fill_halo_size

    # Fill all periodic halos
    Hx, Hy, Hz = halo

    latitude_longitude_grid = LatitudeLongitudeGrid(; size,
                                                      latitude,
                                                      longitude,
                                                      z,
                                                      halo,
                                                      radius)

    # Continue the metrics to the south with the LatitudeLongitudeGrid
    # metrics (probably we don't even need to do this, since the tripolar grid should
    # terminate below Antartica, but it's better to be safe)
    continue_south!(Δxᶠᶠᵃ, latitude_longitude_grid.Δxᶠᶠᵃ)
    continue_south!(Δxᶠᶜᵃ, latitude_longitude_grid.Δxᶠᶜᵃ)
    continue_south!(Δxᶜᶠᵃ, latitude_longitude_grid.Δxᶜᶠᵃ)
    continue_south!(Δxᶜᶜᵃ, latitude_longitude_grid.Δxᶜᶜᵃ)
    
    continue_south!(Δyᶠᶠᵃ, latitude_longitude_grid.Δyᶠᶜᵃ)
    continue_south!(Δyᶠᶜᵃ, latitude_longitude_grid.Δyᶠᶜᵃ)
    continue_south!(Δyᶜᶠᵃ, latitude_longitude_grid.Δyᶜᶠᵃ)
    continue_south!(Δyᶜᶜᵃ, latitude_longitude_grid.Δyᶜᶠᵃ)

    continue_south!(Azᶠᶠᵃ, latitude_longitude_grid.Azᶠᶠᵃ)
    continue_south!(Azᶠᶜᵃ, latitude_longitude_grid.Azᶠᶜᵃ)
    continue_south!(Azᶜᶠᵃ, latitude_longitude_grid.Azᶜᶠᵃ)
    continue_south!(Azᶜᶜᵃ, latitude_longitude_grid.Azᶜᶜᵃ)

    # Final grid with correct metrics
    # TODO: remove `on_architecture(arch, ...)` when we shift grid construction to GPU
    grid = OrthogonalSphericalShellGrid{Periodic, RightConnected, Bounded}(arch,
                                                                           Nx, Ny, Nz,
                                                                           Hx, Hy, Hz,
                                                                           convert(eltype(radius), Lz),
                                                                           on_architecture(arch, λᶜᶜᵃ),
                                                                           on_architecture(arch, λᶠᶜᵃ),
                                                                           on_architecture(arch, λᶜᶠᵃ),
                                                                           on_architecture(arch, λᶠᶠᵃ),
                                                                           on_architecture(arch, φᶜᶜᵃ),
                                                                           on_architecture(arch, φᶠᶜᵃ),
                                                                           on_architecture(arch, φᶜᶠᵃ),
                                                                           on_architecture(arch, φᶠᶠᵃ),
                                                                           on_architecture(arch, zᵃᵃᶜ),
                                                                           on_architecture(arch, zᵃᵃᶠ),
                                                                           on_architecture(arch, Δxᶜᶜᵃ),
                                                                           on_architecture(arch, Δxᶠᶜᵃ),
                                                                           on_architecture(arch, Δxᶜᶠᵃ),
                                                                           on_architecture(arch, Δxᶠᶠᵃ),
                                                                           on_architecture(arch, Δyᶜᶜᵃ),
                                                                           on_architecture(arch, Δyᶜᶠᵃ),
                                                                           on_architecture(arch, Δyᶠᶜᵃ),
                                                                           on_architecture(arch, Δyᶠᶠᵃ),
                                                                           on_architecture(arch, Δzᵃᵃᶜ),
                                                                           on_architecture(arch, Δzᵃᵃᶠ),
                                                                           on_architecture(arch, Azᶜᶜᵃ),
                                                                           on_architecture(arch, Azᶠᶜᵃ),
                                                                           on_architecture(arch, Azᶜᶠᵃ),
                                                                           on_architecture(arch, Azᶠᶠᵃ),
                                                                           radius,
                                                                           Tripolar(north_poles_latitude, first_pole_longitude, southernmost_latitude))
             
    return grid
end

# Continue the metrics to the south with LatitudeLongitudeGrid metrics
function continue_south!(new_metric, lat_lon_metric::Number)
    Hx, Hy = new_metric.offsets
    Nx, Ny = size(new_metric)
    for i in Hx+1:Nx+Hx, j in Hy+1:1
        new_metric[i, j] = lat_lon_metric
    end

    return nothing
end

# Continue the metrics to the south with LatitudeLongitudeGrid metrics
function continue_south!(new_metric, lat_lon_metric::AbstractArray{<:Any, 1})
    Hx, Hy = new_metric.offsets
    Nx, Ny = size(new_metric)
    for i in Hx+1:Nx+Hx, j in Hy+1:1
        new_metric[i, j] = lat_lon_metric[j]
    end

    return nothing
end

# Continue the metrics to the south with LatitudeLongitudeGrid metrics
function continue_south!(new_metric, lat_lon_metric::AbstractArray{<:Any, 2})
    Hx, Hy = - new_metric.offsets
    Nx, Ny = size(new_metric)
    for i in Hx+1:Nx+Hx, j in Hy+1:1
        new_metric[i, j] = lat_lon_metric[i, j]
    end

    return nothing
end

    
#####
##### Outer functions for filling halo regions for Zipper boundary conditions.
#####

@inline function fold_north_face_face!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 2 # Remember! element Nx + 1 does not exist!
    s  = ifelse(i′ > Nx , abs(sign), sign) # for periodic elements we change the sign
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = s * c[i′, Ny - j + 1, k] 
        end
    end

    return nothing
end

#####
##### Fold north functions, used to fold the north direction onto itself
#####

@inline function fold_north_face_center!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 2 # Remember! element Nx + 1 does not exist!
    s  = ifelse(i′ > Nx , abs(sign), sign) # for periodic elements we change the sign
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = s * c[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
        end
    end

    return nothing
end

@inline function fold_north_center_face!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 1
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j + 1, k] 
        end
    end

    return nothing
end

@inline function fold_north_center_center!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 1
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
        end
    end

    return nothing
end


"""
    _compute_tripolar_coordinates!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC, 
                                   λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶠᵃ, φᵃᶜᵃ, 
                                   first_pole_longitude,
                                   focal_distance, Nλ)

Compute the tripolar coordinates for a given set of input parameters. Here, we follow,
the formulation described by

> Ross J. Murray, (1996). Explicit generation of orthogonal grids for ocean models, _Journal of Computational Physics_, **126(2)**, 251-273.

The tripolar grid is built as a set of cofocal ellipsed and perpendicular hyperbolae.
The `focal_distance` argument is the distance from the center of the ellipses to the foci.

The family of ellipses obeys:

```math
\\frac{x²}{a² \\cosh²(ψ)} + \\frac{y²}{a² \\sinh²(ψ)} = 1
```

While the family of perpendicular hyperbolae obey:

```math
\\frac{x²}{a² \\cosh²(λ)} + \\frac{y²}{a² \\sinh²(λ)} = 1
```

Where ``a`` is the `focal_distance` to the center, ``λ`` is the longitudinal angle,
and ``ψ`` is the "isometric latitude", defined by Murray (1996) and satisfying:

```math
    a \\sinh(ψ) = \\mathrm{tand}[(90 - φ) / 2]
```

The final ``(x, y)`` points that define the stereographic projection of the tripolar
coordinates are given by:

```math
    \\begin{align}
    x & = a \\sinh ψ \\cos λ \\\\
    y & = a \\sinh ψ \\sin λ
    \\end{align}
```

for which it is possible to retrieve the longitude and latitude by:

```math
    \\begin{align}
    λ &=    - \\frac{180}{π} \\mathrm{atan}(y / x)  \\\\
    φ &= 90 - \\frac{360}{π} \\mathrm{atan} \\sqrt{x² + y²}
    \\end{align}
```
"""
@kernel function _compute_tripolar_coordinates!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC,
                                                λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                first_pole_longitude,
                                                focal_distance, Nλ)

    i, j = @index(Global, NTuple)

    λ2Ds = (λFF,  λFC,  λCF,  λCC)
    φ2Ds = (φFF,  φFC,  φCF,  φCC)
    λ1Ds = (λᶠᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, λᶜᵃᵃ)
    φ1Ds = (φᵃᶠᵃ, φᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ)

    for (λ2D, φ2D, λ1D, φ1D) in zip(λ2Ds, φ2Ds, λ1Ds, φ1Ds)
        ψ = asinh(tand((90 - φ1D[j]) / 2) / focal_distance)
        x = focal_distance * sind(λ1D[i]) * cosh(ψ)
        y = focal_distance * cosd(λ1D[i]) * sinh(ψ)

        # When x == 0 and y == 0 we are exactly at the north pole,
        # λ (which depends on `atan(y / x)`) is not defined
        # This makes sense, what is the longitude of the north pole? Could be anything!
        # so we choose a value that is continuous with the surrounding points.
        on_the_north_pole = (x == 0) & (y == 0)
        north_pole_value  = ifelse(i == 1, -90, 90) 

        λ2D[i, j] = ifelse(on_the_north_pole, north_pole_value, - 180 / π * atan(y / x))
        φ2D[i, j] = 90 - 360 / π * atan(sqrt(y^2 + x^2)) # The latitude will be in the range [-90, 90]

        # Shift longitude to the range [-180, 180], the 
        # the north singularities will be located at -90 and 90
        λ2D[i, j] += ifelse(i ≤ Nλ÷2, -90, 90) 

        # Make sure the singularities are at longitude we want them to be at.
        # (`first_pole_longitude` and `first_pole_longitude` + 180)
        λ2D[i, j] += first_pole_longitude + 90
        λ2D[i, j]  = convert_to_0_360(λ2D[i, j])
    end
end

# Calculate the metric terms from the coordinates of the grid
# Note: There is probably a better way to do this. Murray (1996) gives
# analytical expressions for the metric terms.
@kernel function _calculate_metrics!(Δxᶠᶜᵃ, Δxᶜᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                     Δyᶠᶜᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                                     Azᶠᶜᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                     λᶠᶜᵃ, λᶜᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                     φᶠᶜᵃ, φᶜᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ, radius)

    i, j = @index(Global, NTuple)

    @inbounds begin
        Δxᶜᶜᵃ[i, j] = haversine((λᶠᶜᵃ[i+1, j], φᶠᶜᵃ[i+1, j]), (λᶠᶜᵃ[i, j],   φᶠᶜᵃ[i, j]),   radius)
        Δxᶠᶜᵃ[i, j] = haversine((λᶜᶜᵃ[i, j],   φᶜᶜᵃ[i, j]),   (λᶜᶜᵃ[i-1, j], φᶜᶜᵃ[i-1, j]), radius)
        Δxᶜᶠᵃ[i, j] = haversine((λᶠᶠᵃ[i+1, j], φᶠᶠᵃ[i+1, j]), (λᶠᶠᵃ[i, j],   φᶠᶠᵃ[i, j]),   radius) 
        Δxᶠᶠᵃ[i, j] = haversine((λᶜᶠᵃ[i, j],   φᶜᶠᵃ[i, j]),   (λᶜᶠᵃ[i-1, j], φᶜᶠᵃ[i-1, j]), radius)

        Δyᶜᶜᵃ[i, j] = haversine((λᶜᶠᵃ[i, j+1], φᶜᶠᵃ[i, j+1]),   (λᶜᶠᵃ[i, j],   φᶜᶠᵃ[i, j]),   radius)
        Δyᶠᶜᵃ[i, j] = haversine((λᶠᶠᵃ[i, j+1], φᶠᶠᵃ[i, j+1]),   (λᶠᶠᵃ[i, j],   φᶠᶠᵃ[i, j]),   radius)
        Δyᶜᶠᵃ[i, j] = haversine((λᶜᶜᵃ[i, j  ],   φᶜᶜᵃ[i, j]),   (λᶜᶜᵃ[i, j-1], φᶜᶜᵃ[i, j-1]), radius)
        Δyᶠᶠᵃ[i, j] = haversine((λᶠᶜᵃ[i, j  ],   φᶠᶜᵃ[i, j]),   (λᶠᶜᵃ[i, j-1], φᶠᶜᵃ[i, j-1]), radius)

        a = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)
        b = lat_lon_to_cartesian(φᶠᶠᵃ[i+1,  j ], λᶠᶠᵃ[i+1,  j ], 1)
        c = lat_lon_to_cartesian(φᶠᶠᵃ[i+1, j+1], λᶠᶠᵃ[i+1, j+1], 1)
        d = lat_lon_to_cartesian(φᶠᶠᵃ[ i , j+1], λᶠᶠᵃ[ i , j+1], 1)

        Azᶜᶜᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2

        # To be able to conserve kinetic energy specifically the momentum equation, 
        # it is better to define the face areas as products of
        # the edge lengths rather than using the spherical area of the face (cit JMC).
        # TODO: find a reference to support this statement
        Azᶠᶜᵃ[i, j] = Δyᶠᶜᵃ[i, j] * Δxᶠᶜᵃ[i, j]
        Azᶜᶠᵃ[i, j] = Δyᶜᶠᵃ[i, j] * Δxᶜᶠᵃ[i, j]

        # Face - Face areas are calculated as the Center - Center ones
        a = lat_lon_to_cartesian(φᶜᶜᵃ[i-1, j-1], λᶜᶜᵃ[i-1, j-1], 1)
        b = lat_lon_to_cartesian(φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1], 1)
        c = lat_lon_to_cartesian(φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ], 1)
        d = lat_lon_to_cartesian(φᶜᶜᵃ[i-1,  j ], λᶜᶜᵃ[i-1,  j ], 1)

        Azᶠᶠᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2 
    end
end

function with_halo(new_halo, old_grid::TripolarGrid)

    size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)

    z = cpu_face_constructor_z(old_grid)

    north_poles_latitude = old_grid.conformal_mapping.north_poles_latitude
    first_pole_longitude = old_grid.conformal_mapping.first_pole_longitude
    southernmost_latitude = old_grid.conformal_mapping.southernmost_latitude

    new_grid = TripolarGrid(architecture(old_grid), eltype(old_grid);
                            size, z, halo = new_halo,
                            radius = old_grid.radius,
                            north_poles_latitude,
                            first_pole_longitude,
                            southernmost_latitude)

    return new_grid
end
