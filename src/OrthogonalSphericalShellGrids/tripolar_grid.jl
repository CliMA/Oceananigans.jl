using Oceananigans.BoundaryConditions: ZipperBoundaryCondition
using Oceananigans.Grids: architecture, cpu_face_constructor_z

import Oceananigans.Grids: with_halo, validate_dimension_specification

"""
    struct Tripolar{N, F, S}

A structure to represent a tripolar grid on a spherical shell.
"""
struct Tripolar{N, F, S}
    north_poles_latitude :: N
    first_pole_longitude :: F
    southernmost_latitude :: S
end

Adapt.adapt_structure(to, t::Tripolar) =
    Tripolar(Adapt.adapt(to, t.north_poles_latitude),
             Adapt.adapt(to, t.first_pole_longitude),
             Adapt.adapt(to, t.southernmost_latitude))

const TripolarGrid{FT, TX, TY, TZ, CZ, CC, FC, CF, FF, Arch} = OrthogonalSphericalShellGrid{FT, TX, TY, TZ, CZ, <:Tripolar, CC, FC, CF, FF, Arch}
const TripolarGridOfSomeKind = Union{TripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}}

"""
    TripolarGrid(arch = CPU(), FT::DataType = Float64;
                 size,
                 southernmost_latitude = -80,
                 halo = (4, 4, 4),
                 radius = R_Earth,
                 z = (0, 1),
                 north_poles_latitude = 55,
                 first_pole_longitude = 70)

Return an `OrthogonalSphericalShellGrid` tripolar grid on the sphere. The
tripolar grid replaces the North pole singularity with two other singularities
at `north_poles_latitude` that is _less_ than 90ᵒ.

Positional Arguments
====================

- `arch`: The architecture to use for the grid. Default is `CPU()`.
- `FT::DataType`: The data type to use for the grid. Default is `Float64`.

Keyword Arguments
=================

- `size`: The number of cells in the (longitude, latitude, vertical) dimensions.
- `southernmost_latitude`: The southernmost `Center` latitude of the grid. Default: -80.
- `halo`: The halo size in the (longitude, latitude, vertical) dimensions. Default: (4, 4, 4).
- `radius`: The radius of the spherical shell. Default: `R_Earth`.
- `z`: The vertical ``z``-coordinate range of the grid. Could either be (i) 2-tuple that specifies
       the end points of the coordinate, (ii) an array with the ``z`` interfaces, or (iii) a function
       of `k` index that returns the locations of cell interfaces in ``z``-direction. Default: (0, 1).
- `first_pole_longitude`: The longitude of the first "north" singularity.
                          The second singularity is located at `first_pole_longitude + 180ᵒ`.
- `north_poles_latitude`: The latitude of the "north" singularities.

!!! warning "Longitude coordinate must have even number of cells"
    `size` is a 3-tuple of the grid size in longitude, latitude, and vertical directions.
    Due to requirements of the folding at the north edge of the domain, the longitude size
    of the grid (i.e., the first component of `size`) _must_ be an even number!

!!! info "North pole singularities"
    The north singularities are located at: `i = 1`, `j = Nφ` and `i = Nλ ÷ 2 + 1`, `j = Nφ`.
"""
function TripolarGrid(arch = CPU(), FT::DataType = Float64;
                      size,
                      southernmost_latitude = -80,
                      halo = (4, 4, 4),
                      radius = R_Earth,
                      z = (0, 1),
                      north_poles_latitude = 55,
                      first_pole_longitude = 70)  # second pole is at longitude `first_pole_longitude + 180ᵒ`

    # TODO: Change a couple of allocations here and there to be able
    # to construct the grid on the GPU. This is not a huge problem as
    # grid generation is quite fast, but it might become slow for
    # sub-kilometer resolution grids.
    latitude  = (southernmost_latitude, 90)
    longitude = (-180, 180)

    focal_distance = tand((90 - north_poles_latitude) / 2)

    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo

    if isodd(Nλ)
        throw(ArgumentError("The number of cells in the longitude dimension should be even!"))
    end

    # the λ and z coordinate is the same as for the other grids,
    # but for the φ coordinate we need to remove one point at the north
    # because the the north pole is a `Center`point, not on `Face` point...
    topology  = (Periodic, RightConnected, Bounded)
    TZ = topology[3]
    z = validate_dimension_specification(TZ, z, :z, Nz, FT)

    Lx, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, topology, size, halo, longitude, :longitude, 1, CPU())
    Lz, z                        = generate_coordinate(FT, topology, size, halo, z,         :z,         3, CPU())

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

    # We need to circshift everything to have the first pole at the beginning of the
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

    # return λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC
    # Helper grid to fill halo
    grid = RectilinearGrid(; size = (Nx, Ny),
                             halo = (Hλ, Hφ),
                             x = (0, 1), y = (0, 1),
                             topology = (Periodic, RightConnected, Flat))

    # Boundary conditions to fill halos of the coordinate and metric terms
    # We need to define them manually because of the convention in the
    # ZipperBoundaryCondition that edge fields need to switch sign (which we definitely do not
    # want for coordinates and metrics)
    default_boundary_conditions = FieldBoundaryConditions(north  = ZipperBoundaryCondition(),
                                                          south  = NoFluxBoundaryCondition(), # The south should be `continued`
                                                          west   = Oceananigans.PeriodicBoundaryCondition(),
                                                          east   = Oceananigans.PeriodicBoundaryCondition(),
                                                          top    = nothing,
                                                          bottom = nothing)

    lFF = Field{Face, Face, Center}(grid; boundary_conditions = default_boundary_conditions)
    pFF = Field{Face, Face, Center}(grid; boundary_conditions = default_boundary_conditions)

    lFC = Field{Face, Center, Center}(grid; boundary_conditions = default_boundary_conditions)
    pFC = Field{Face, Center, Center}(grid; boundary_conditions = default_boundary_conditions)

    lCF = Field{Center, Face, Center}(grid; boundary_conditions = default_boundary_conditions)
    pCF = Field{Center, Face, Center}(grid; boundary_conditions = default_boundary_conditions)

    lCC = Field{Center, Center, Center}(grid; boundary_conditions = default_boundary_conditions)
    pCC = Field{Center, Center, Center}(grid; boundary_conditions = default_boundary_conditions)

    set!(lFF, λFF)
    set!(pFF, φFF)

    set!(lFC, λFC)
    set!(pFC, φFC)

    set!(lCF, λCF)
    set!(pCF, φCF)

    set!(lCC, λCC)
    set!(pCC, φCC)

    fill_halo_regions!(lFF)
    fill_halo_regions!(lCF)
    fill_halo_regions!(lFC)
    fill_halo_regions!(lCC)

    fill_halo_regions!(pFF)
    fill_halo_regions!(pCF)
    fill_halo_regions!(pFC)
    fill_halo_regions!(pCC)

    # Coordinates
    λᶠᶠᵃ = dropdims(lFF.data, dims=3)
    φᶠᶠᵃ = dropdims(pFF.data, dims=3)

    λᶠᶜᵃ = dropdims(lFC.data, dims=3)
    φᶠᶜᵃ = dropdims(pFC.data, dims=3)

    λᶜᶠᵃ = dropdims(lCF.data, dims=3)
    φᶜᶠᵃ = dropdims(pCF.data, dims=3)

    λᶜᶜᵃ = dropdims(lCC.data, dims=3)
    φᶜᶜᵃ = dropdims(pCC.data, dims=3)

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

    # Metrics fields to fill halos
    FF = Field{Face,   Face,   Center}(grid; boundary_conditions = default_boundary_conditions)
    FC = Field{Face,   Center, Center}(grid; boundary_conditions = default_boundary_conditions)
    CF = Field{Center, Face,   Center}(grid; boundary_conditions = default_boundary_conditions)
    CC = Field{Center, Center, Center}(grid; boundary_conditions = default_boundary_conditions)

    # Fill all periodic halos
    set!(FF, Δxᶠᶠᵃ)
    set!(CF, Δxᶜᶠᵃ)
    set!(FC, Δxᶠᶜᵃ)
    set!(CC, Δxᶜᶜᵃ)
    fill_halo_regions!(FF)
    fill_halo_regions!(CF)
    fill_halo_regions!(FC)
    fill_halo_regions!(CC)
    Δxᶠᶠᵃ = deepcopy(dropdims(FF.data, dims=3))
    Δxᶜᶠᵃ = deepcopy(dropdims(CF.data, dims=3))
    Δxᶠᶜᵃ = deepcopy(dropdims(FC.data, dims=3))
    Δxᶜᶜᵃ = deepcopy(dropdims(CC.data, dims=3))

    set!(FF, Δyᶠᶠᵃ)
    set!(CF, Δyᶜᶠᵃ)
    set!(FC, Δyᶠᶜᵃ)
    set!(CC, Δyᶜᶜᵃ)
    fill_halo_regions!(FF)
    fill_halo_regions!(CF)
    fill_halo_regions!(FC)
    fill_halo_regions!(CC)
    Δyᶠᶠᵃ = deepcopy(dropdims(FF.data, dims=3))
    Δyᶜᶠᵃ = deepcopy(dropdims(CF.data, dims=3))
    Δyᶠᶜᵃ = deepcopy(dropdims(FC.data, dims=3))
    Δyᶜᶜᵃ = deepcopy(dropdims(CC.data, dims=3))

    set!(FF, Azᶠᶠᵃ)
    set!(CF, Azᶜᶠᵃ)
    set!(FC, Azᶠᶜᵃ)
    set!(CC, Azᶜᶜᵃ)
    fill_halo_regions!(FF)
    fill_halo_regions!(CF)
    fill_halo_regions!(FC)
    fill_halo_regions!(CC)
    Azᶠᶠᵃ = deepcopy(dropdims(FF.data, dims=3))
    Azᶜᶠᵃ = deepcopy(dropdims(CF.data, dims=3))
    Azᶠᶜᵃ = deepcopy(dropdims(FC.data, dims=3))
    Azᶜᶜᵃ = deepcopy(dropdims(CC.data, dims=3))

    Hx, Hy, Hz = halo

    latitude_longitude_grid = LatitudeLongitudeGrid(; size,
                                                      latitude,
                                                      longitude,
                                                      halo,
                                                      z = (0, 1), # z does not really matter here
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
                                                                           convert(FT, Lz),
                                                                           on_architecture(arch, map(FT, λᶜᶜᵃ)),
                                                                           on_architecture(arch, map(FT, λᶠᶜᵃ)),
                                                                           on_architecture(arch, map(FT, λᶜᶠᵃ)),
                                                                           on_architecture(arch, map(FT, λᶠᶠᵃ)),
                                                                           on_architecture(arch, map(FT, φᶜᶜᵃ)),
                                                                           on_architecture(arch, map(FT, φᶠᶜᵃ)),
                                                                           on_architecture(arch, map(FT, φᶜᶠᵃ)),
                                                                           on_architecture(arch, map(FT, φᶠᶠᵃ)),
                                                                           on_architecture(arch, z),
                                                                           on_architecture(arch, map(FT, Δxᶜᶜᵃ)),
                                                                           on_architecture(arch, map(FT, Δxᶠᶜᵃ)),
                                                                           on_architecture(arch, map(FT, Δxᶜᶠᵃ)),
                                                                           on_architecture(arch, map(FT, Δxᶠᶠᵃ)),
                                                                           on_architecture(arch, map(FT, Δyᶜᶜᵃ)),
                                                                           on_architecture(arch, map(FT, Δyᶠᶜᵃ)),
                                                                           on_architecture(arch, map(FT, Δyᶜᶠᵃ)),
                                                                           on_architecture(arch, map(FT, Δyᶠᶠᵃ)),
                                                                           on_architecture(arch, map(FT, Azᶜᶜᵃ)),
                                                                           on_architecture(arch, map(FT, Azᶠᶜᵃ)),
                                                                           on_architecture(arch, map(FT, Azᶜᶠᵃ)),
                                                                           on_architecture(arch, map(FT, Azᶠᶠᵃ)),
                                                                           convert(FT, radius),
                                                                           Tripolar(north_poles_latitude, first_pole_longitude, southernmost_latitude))

    return grid
end

# Continue the metrics to the south with LatitudeLongitudeGrid metrics
function continue_south!(new_metric, lat_lon_metric::Number)
    Hx, Hy = new_metric.offsets
    Nx, Ny = size(new_metric)

    for j in Hy+1:1, i in Hx+1:Nx+Hx
        @inbounds new_metric[i, j] = lat_lon_metric
    end

    return nothing
end

# Continue the metrics to the south with LatitudeLongitudeGrid metrics
function continue_south!(new_metric, lat_lon_metric::AbstractArray{<:Any, 1})
    Hx, Hy = new_metric.offsets
    Nx, Ny = size(new_metric)

    for j in Hy+1:1, i in Hx+1:Nx+Hx
        @inbounds new_metric[i, j] = lat_lon_metric[j]
    end

    return nothing
end

# Continue the metrics to the south with LatitudeLongitudeGrid metrics
function continue_south!(new_metric, lat_lon_metric::AbstractArray{<:Any, 2})
    Hx, Hy = - new_metric.offsets
    Nx, Ny = size(new_metric)

    for j in Hy+1:1, i in Hx+1:Nx+Hx
        @inbounds new_metric[i, j] = lat_lon_metric[i, j]
    end

    return nothing
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
