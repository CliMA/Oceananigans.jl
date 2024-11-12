using Distances
using CUDA: @allowscalar
using Oceananigans.Architectures: device

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

    data_args = ((Periodic, Bounded, Bounded), size, halo)

    # Start with the NH stereographic projection
    # TODO: make these on_architecture(arch, zeros(Nx, Ny))
    # to build the grid on GPU
    λᶜᶜᵃ = new_data(FT, arch, (Center, Center, Nothing), data_args...)
    λᶠᶜᵃ = new_data(FT, arch, (Face,   Center, Nothing), data_args...)
    λᶜᶠᵃ = new_data(FT, arch, (Center, Face,   Nothing), data_args...)
    λᶠᶠᵃ = new_data(FT, arch, (Face,   Face,   Nothing), data_args...)
    φᶜᶜᵃ = new_data(FT, arch, (Center, Center, Nothing), data_args...)
    φᶠᶜᵃ = new_data(FT, arch, (Face,   Center, Nothing), data_args...)
    φᶜᶠᵃ = new_data(FT, arch, (Center, Face,   Nothing), data_args...)
    φᶠᶠᵃ = new_data(FT, arch, (Face,   Face,   Nothing), data_args...)

    coords     = (λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ, φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ)
    coords_llg = (λᶜᵃᵃ, λᶠᵃᵃ, φᵃᶜᵃ, φᵃᶠᵃ)

    _compute_tripolar_coordinates!(device(arch), (16, 16), (Nλ, Nφ))(coords..., coords_llg...,
                                                                     first_pole_longitude,
                                                                     focal_distance, Nλ)

    # We need to circshift eveything to have the first pole at the beginning of the 
    # grid and the second pole in the middle which is necessary for the folding
    # at the north edge of the domain
    shift = Base.size(λᶜᶜᵃ, 1) ÷ 4

    @show shift
    λᶜᶜᵃ = @allowscalar circshift(λᶜᶜᵃ, (shift, 0, 0)) 
    λᶠᶜᵃ = @allowscalar circshift(λᶠᶜᵃ, (shift, 0, 0))
    λᶜᶠᵃ = @allowscalar circshift(λᶜᶠᵃ, (shift, 0, 0))
    λᶠᶠᵃ = @allowscalar circshift(λᶠᶠᵃ, (shift, 0, 0)) 
    φᶜᶜᵃ = @allowscalar circshift(φᶜᶜᵃ, (shift, 0, 0)) 
    φᶠᶜᵃ = @allowscalar circshift(φᶠᶜᵃ, (shift, 0, 0)) 
    φᶜᶠᵃ = @allowscalar circshift(φᶜᶠᵃ, (shift, 0, 0)) 
    φᶠᶠᵃ = @allowscalar circshift(φᶠᶠᵃ, (shift, 0, 0)) 

    # Metrics fields to fill fill_halo_size
    coords_x_loc = (Center(), Face(), Center(), Face(), Center(), Face(), Center(), Face())
    coords_y_loc = (Center(), Center(), Face(), Face(), Center(), Center(), Face(), Face())

    for (coord, ℓx, ℓy) in zip(coords, coords_x_loc, coords_y_loc)
        _fold_tripolar_metrics!(device(arch), 16, Nλ)(coord, ℓx, ℓy, Nλ, Nφ, Hλ, Hφ)
        _fill_periodic_metric_halos!(device(arch), 16, Nφ)(coord, Nλ, Hλ, Hφ)
    end

    λᶜᶜᵃ = dropdims(λᶜᶜᵃ, dims = 3)
    λᶠᶜᵃ = dropdims(λᶠᶜᵃ, dims = 3)
    λᶜᶠᵃ = dropdims(λᶜᶠᵃ, dims = 3)
    λᶠᶠᵃ = dropdims(λᶠᶠᵃ, dims = 3)
    φᶜᶜᵃ = dropdims(φᶜᶜᵃ, dims = 3)
    φᶠᶜᵃ = dropdims(φᶠᶜᵃ, dims = 3)
    φᶜᶠᵃ = dropdims(φᶜᶠᵃ, dims = 3)
    φᶠᶠᵃ = dropdims(φᶠᶠᵃ, dims = 3)

    # Allocate Metrics
    Δxᶜᶜᵃ = new_data(FT, arch, (Face,   Face,   Nothing), data_args...)
    Δxᶠᶜᵃ = new_data(FT, arch, (Face,   Center, Nothing), data_args...)
    Δxᶜᶠᵃ = new_data(FT, arch, (Center, Face,   Nothing), data_args...)
    Δxᶠᶠᵃ = new_data(FT, arch, (Center, Center, Nothing), data_args...)     
    Δyᶜᶜᵃ = new_data(FT, arch, (Face,   Face,   Nothing), data_args...)
    Δyᶠᶜᵃ = new_data(FT, arch, (Face,   Center, Nothing), data_args...)
    Δyᶜᶠᵃ = new_data(FT, arch, (Center, Face,   Nothing), data_args...)
    Δyᶠᶠᵃ = new_data(FT, arch, (Center, Center, Nothing), data_args...)
    Azᶜᶜᵃ = new_data(FT, arch, (Face,   Face,   Nothing), data_args...)
    Azᶠᶜᵃ = new_data(FT, arch, (Face,   Center, Nothing), data_args...)
    Azᶜᶠᵃ = new_data(FT, arch, (Center, Face,   Nothing), data_args...)
    Azᶠᶠᵃ = new_data(FT, arch, (Center, Center, Nothing), data_args...)

    metrics = (Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
               Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
               Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ)

    _calculate_metrics!(device(arch), (16, 16), (Nλ, Nφ))(metrics..., coords..., radius)

    metrics_x_loc = (Center(), Face(), Center(), Face(), Center(), Face(), Center(), Face(), Center(), Face(), Center(), Face())
    metrics_y_loc = (Center(), Center(), Face(), Face(), Center(), Center(), Face(), Face(), Center(), Center(), Face(), Face())

    for (metric, ℓx, ℓy) in zip(metrics, metrics_x_loc, metrics_y_loc)
        _fold_tripolar_metrics!(device(arch), 16, Nλ)(metric, ℓx, ℓy, Nλ, Nφ, Hλ, Hφ)
        _fill_periodic_metric_halos!(device(arch), 16, Nφ)(metric, Nλ, Hλ, Hφ)
    end

    Δxᶜᶜᵃ = dropdims(Δxᶜᶜᵃ, dims = 3)
    Δxᶠᶜᵃ = dropdims(Δxᶠᶜᵃ, dims = 3)
    Δxᶜᶠᵃ = dropdims(Δxᶜᶠᵃ, dims = 3)
    Δxᶠᶠᵃ = dropdims(Δxᶠᶠᵃ, dims = 3)
    Δyᶜᶜᵃ = dropdims(Δyᶜᶜᵃ, dims = 3)
    Δyᶠᶜᵃ = dropdims(Δyᶠᶜᵃ, dims = 3)
    Δyᶜᶠᵃ = dropdims(Δyᶜᶠᵃ, dims = 3)
    Δyᶠᶠᵃ = dropdims(Δyᶠᶠᵃ, dims = 3)
    Azᶜᶜᵃ = dropdims(Azᶜᶜᵃ, dims = 3)
    Azᶠᶜᵃ = dropdims(Azᶠᶜᵃ, dims = 3)
    Azᶜᶠᵃ = dropdims(Azᶜᶠᵃ, dims = 3)
    Azᶠᶠᵃ = dropdims(Azᶠᶠᵃ, dims = 3)

    conformal_map = Tripolar(north_poles_latitude, first_pole_longitude, southernmost_latitude)

    # Final grid with correct metrics
    return OrthogonalSphericalShellGrid{Periodic, RightConnected, Bounded}(arch,
                                                                           Nλ, Nφ, Nz,
                                                                           Hλ, Hφ, Hz,
                                                                           convert(eltype(radius), Lz),
                                                                           λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                                                           φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ,
                                                                           zᵃᵃᶜ, zᵃᵃᶠ,
                                                                           Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                                           Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ,
                                                                           Δzᵃᵃᶜ, Δzᵃᵃᶠ, 
                                                                           Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                                                           radius,
                                                                           conformal_map)
end

#####
##### Fold north functions, used to fold the north direction onto itself
#####

@kernel function _fill_periodic_metric_halos!(metric, Nx, Hx, Hy)    
    j  = @index(Global, Linear)
    j′ = j - Hy 

    # Fill periodic halos:
    for i = 1 : Hx
        metric[1  - i, j′, 1] = metric[Nx - i + 1, j′, 1]
        metric[Nx + i, j′, 1] = metric[i,          j′, 1]
    end
end

@kernel function _fold_tripolar_metrics!(metric, ℓx, ℓy, Nx, Ny, Hx, Hy)
    i = @index(Global, Linear)
    fold_north_boundary!(metric, i, 1, ℓx, ℓy, Nx, Ny, Hy, 1)
end

@inline function fold_north_boundary!(c, i, k, ::Center, ::Center, Nx, Ny, Hy, sign)    
    i′ = Nx - i + 1
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
        end
    end

    return nothing
end

@inline function fold_north_boundary!(c, i, k, ::Face, ::Center, Nx, Ny, Hy, sign)    
    i′ = Nx - i + 2 # Remember! element Nx + 1 does not exist!
    s  = ifelse(i′ > Nx , abs(sign), sign) # for periodic elements we change the sign
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = s * c[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
        end
    end

    return nothing
end

@inline function fold_north_boundary!(c, i, k, ::Center, ::Face, Nx, Ny, Hy, sign)    
    i′ = Nx - i + 1
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j + 1, k] 
        end
    end

    return nothing
end

@inline function fold_north_boundary!(c, i, k, ::Face, ::Face, Nx, Ny, Hy, sign)    
    i′ = Nx - i + 2 # Remember! element Nx + 1 does not exist!
    s  = ifelse(i′ > Nx , abs(sign), sign) # for periodic elements we change the sign
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = s * c[i′, Ny - j + 1, k] 
        end
    end

    return nothing
end

"""
    _compute_tripolar_coordinates!(λCC, λFC, λCF, λFF, φCC, φFC, φCF, φFF, 
                                   λᶜᵃᵃ, λᶠᵃᵃ, φᵃᶜᵃ, φᵃᶠᵃ, 
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
@kernel function _compute_tripolar_coordinates!(λCC, λFC, λCF, λFF, φCC, φFC, φCF, φFF, 
                                                λᶜᵃᵃ, λᶠᵃᵃ, φᵃᶜᵃ, φᵃᶠᵃ,
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

        λ2D[i, j, 1] = ifelse(on_the_north_pole, north_pole_value, - 180 / π * atan(y / x))
        φ2D[i, j, 1] = 90 - 360 / π * atan(sqrt(y^2 + x^2)) # The latitude will be in the range [-90, 90]

        # Shift longitude to the range [-180, 180], the 
        # the north singularities will be located at -90 and 90
        λ2D[i, j, 1] += ifelse(i ≤ Nλ÷2, -90, 90) 

        # Make sure the singularities are at longitude we want them to be at.
        # (`first_pole_longitude` and `first_pole_longitude` + 180)
        λ2D[i, j, 1] += first_pole_longitude + 90
        λ2D[i, j, 1]  = convert_to_0_360(λ2D[i, j])
    end
end

# Calculate the metric terms from the coordinates of the grid
# Note: There is probably a better way to do this. Murray (1996) gives
# analytical expressions for the metric terms.
@kernel function _calculate_metrics!(Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                     Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                                     Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                     λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                     φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ, radius)

    i, j = @index(Global, NTuple)

    @inbounds begin
        Δxᶜᶜᵃ[i, j, 1] = haversine((λᶠᶜᵃ[i+1, j], φᶠᶜᵃ[i+1, j]), (λᶠᶜᵃ[i, j],   φᶠᶜᵃ[i, j]),   radius)
        Δxᶠᶜᵃ[i, j, 1] = haversine((λᶜᶜᵃ[i, j],   φᶜᶜᵃ[i, j]),   (λᶜᶜᵃ[i-1, j], φᶜᶜᵃ[i-1, j]), radius)
        Δxᶜᶠᵃ[i, j, 1] = haversine((λᶠᶠᵃ[i+1, j], φᶠᶠᵃ[i+1, j]), (λᶠᶠᵃ[i, j],   φᶠᶠᵃ[i, j]),   radius) 
        Δxᶠᶠᵃ[i, j, 1] = haversine((λᶜᶠᵃ[i, j],   φᶜᶠᵃ[i, j]),   (λᶜᶠᵃ[i-1, j], φᶜᶠᵃ[i-1, j]), radius)

        Δyᶜᶜᵃ[i, j, 1] = haversine((λᶜᶠᵃ[i, j+1], φᶜᶠᵃ[i, j+1]),   (λᶜᶠᵃ[i, j],   φᶜᶠᵃ[i, j]),   radius)
        Δyᶠᶜᵃ[i, j, 1] = haversine((λᶠᶠᵃ[i, j+1], φᶠᶠᵃ[i, j+1]),   (λᶠᶠᵃ[i, j],   φᶠᶠᵃ[i, j]),   radius)
        Δyᶜᶠᵃ[i, j, 1] = haversine((λᶜᶜᵃ[i, j  ],   φᶜᶜᵃ[i, j]),   (λᶜᶜᵃ[i, j-1], φᶜᶜᵃ[i, j-1]), radius)
        Δyᶠᶠᵃ[i, j, 1] = haversine((λᶠᶜᵃ[i, j  ],   φᶠᶜᵃ[i, j]),   (λᶠᶜᵃ[i, j-1], φᶠᶜᵃ[i, j-1]), radius)

        a = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)
        b = lat_lon_to_cartesian(φᶠᶠᵃ[i+1,  j ], λᶠᶠᵃ[i+1,  j ], 1)
        c = lat_lon_to_cartesian(φᶠᶠᵃ[i+1, j+1], λᶠᶠᵃ[i+1, j+1], 1)
        d = lat_lon_to_cartesian(φᶠᶠᵃ[ i , j+1], λᶠᶠᵃ[ i , j+1], 1)

        Azᶜᶜᵃ[i, j, 1] = spherical_area_quadrilateral(a, b, c, d) * radius^2

        # To be able to conserve kinetic energy specifically the momentum equation, 
        # it is better to define the face areas as products of
        # the edge lengths rather than using the spherical area of the face (cit JMC).
        # TODO: find a reference to support this statement
        Azᶠᶜᵃ[i, j, 1] = Δyᶠᶜᵃ[i, j, 1] * Δxᶠᶜᵃ[i, j, 1]
        Azᶜᶠᵃ[i, j, 1] = Δyᶜᶠᵃ[i, j, 1] * Δxᶜᶠᵃ[i, j, 1]

        # Face - Face areas are calculated as the Center - Center ones
        a = lat_lon_to_cartesian(φᶜᶜᵃ[i-1, j-1], λᶜᶜᵃ[i-1, j-1], 1)
        b = lat_lon_to_cartesian(φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1], 1)
        c = lat_lon_to_cartesian(φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ], 1)
        d = lat_lon_to_cartesian(φᶜᶜᵃ[i-1,  j ], λᶜᶜᵃ[i-1,  j ], 1)

        Azᶠᶠᵃ[i, j, 1] = spherical_area_quadrilateral(a, b, c, d) * radius^2 
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
