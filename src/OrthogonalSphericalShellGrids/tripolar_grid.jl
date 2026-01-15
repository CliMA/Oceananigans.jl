using Oceananigans.BoundaryConditions: UPivotZipperBoundaryCondition, FPivotZipperBoundaryCondition, NoFluxBoundaryCondition
using Oceananigans.Fields: set!
using Oceananigans.Grids: Grids, Bounded, Flat, OrthogonalSphericalShellGrid, Periodic, RectilinearGrid,
    architecture, cpu_face_constructor_z, validate_dimension_specification,
    RightCenterFolded, RightFaceFolded
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

"""
    struct Tripolar{N, F, S}

A structure to represent a tripolar grid on an orthogonal spherical shell.
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
                 radius = Oceananigans.defaults.planet_radius,
                 z = (0, 1),
                 north_poles_latitude = 55,
                 first_pole_longitude = 70,
                 fold_topology = RightCenterFolded)

Return an `OrthogonalSphericalShellGrid` tripolar grid on the sphere. The
tripolar grid replaces the North pole singularity with two other singularities
at `north_poles_latitude` that is _less_ than 90ᵒ.

The grid is constructed following the formulation by [Murray (1996)](@cite Murray1996).

Positional Arguments
====================

- `arch`: The architecture to use for the grid. Default is `CPU()`.
- `FT::DataType`: The data type to use for the grid. Default is `Float64`.

Keyword Arguments
=================

- `size`: The number of cells in the (longitude, latitude, vertical) dimensions.
- `southernmost_latitude`: The southernmost `Center` latitude of the grid. Default: -80.
- `halo`: The halo size in the (longitude, latitude, vertical) dimensions. Default: (4, 4, 4).
- `radius`: The radius of the spherical shell. Default: `Oceananigans.defaults.planet_radius`.
- `z`: The vertical ``z``-coordinate range of the grid. Could either be:
       (i) 2-tuple that specifies the end points of the coordinate,
       (ii) an array with the ``z`` interfaces, or
       (iii) a function of `k` index that returns the locations of cell interfaces
             in ``z``-direction. Default: (0, 1).
- `first_pole_longitude`: The longitude of the first "north" singularity.
                          The second singularity is located at `first_pole_longitude + 180ᵒ`.
                          Default: 75.
- `north_poles_latitude`: The latitude of the "north" singularities. Default: 55.
- `fold_topology`: The folding topology to use. Either `RightCenterFolded` or `RightFaceFolded`:
    - `RightCenterFolded` folds the north boundary along cell `XFace`s and `Center`s,
        with a pivot point located on a `XFace`.
    - `RightFaceFolded` corresponds to folding the north boundary along `YFace`s,
        with a pivot point located on a corner location `(Face, Face)`.
        Default: `RightCenterFolded`.

!!! warning "Longitude coordinate must have even number of cells"
    `size` is a 3-tuple of the grid size in longitude, latitude, and vertical directions.
    Due to requirements of the folding at the north edge of the domain, the longitude size
    of the grid (i.e., the first component of `size`) _must_ be an even number!

!!! info "North pole singularities"
    The north singularities are located at: `i = 1`, `j = grid.Ny` and
    `i = grid.Nx ÷ 2 + 1`, `j = grid.Ny`.

References
==========

Murray, R. J. (1996). Explicit generation of orthogonal grids for ocean models.
    Journal of Computational Physics, 126(2), 251-273.
"""
function TripolarGrid(arch = CPU(), FT::DataType = Float64;
                      size,
                      southernmost_latitude = -80,
                      halo = (4, 4, 4),
                      radius = Oceananigans.defaults.planet_radius,
                      z = (0, 1),
                      north_poles_latitude = 55,
                      first_pole_longitude = 70, # second pole is at longitude `first_pole_longitude + 180ᵒ`
                      fold_topology = RightCenterFolded)

    # Set the topology
    topology = (Periodic, fold_topology, Bounded)

    # TODO: Change a couple of allocations here and there to be able
    # to construct the grid on the GPU. This is not a huge problem as
    # grid generation is quite fast, but it might become slow for
    # sub-kilometer resolution grids.
    latitude  = (southernmost_latitude, 90)
    longitude = (-180, 180)

    focal_distance = tand((90 - north_poles_latitude) / 2)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    if isodd(Nx)
        throw(ArgumentError("The number of cells in the longitude dimension should be even!"))
    end

    # Generate coordinates
    TZ = topology[3]
    z = validate_dimension_specification(TZ, z, :z, Nz, FT)
    Lz, z                        = generate_coordinate(FT, topology, size, halo, z,         :z,         3, CPU())
    Ly, φᵃᶠᵃ, φᵃᶜᵃ, Δφᶠᵃᵃ, Δφᶜᵃᵃ = generate_coordinate(FT, topology, size, halo, latitude,  :latitude,  2, CPU())
    Lx, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, topology, size, halo, longitude, :longitude, 1, CPU())


    # return λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC
    # Helper grid to lauch kernels on
    grid = RectilinearGrid(; size = (Nx, Ny),
                             halo = (Hx, Hy),
                             x = (0, 1), y = (0, 1),
                             topology = (Periodic, fold_topology, Flat))

    #  Place the fields on the grid
    λFF = Field{Face, Face, Center}(grid)
    φFF = Field{Face, Face, Center}(grid)
    λFC = Field{Face, Center, Center}(grid)
    φFC = Field{Face, Center, Center}(grid)
    λCF = Field{Center, Face, Center}(grid)
    φCF = Field{Center, Face, Center}(grid)
    λCC = Field{Center, Center, Center}(grid)
    φCC = Field{Center, Center, Center}(grid)

    # Compute coordinates using the same kernel twice but with varying size,
    # as the size of λᵃᶠᵃ and φᵃᶠᵃ may vary with the fold topology.
    # Note: we don't fill_halo_regions! and, instead,
    # compute the full fields including in the halos (less code!).
    kp = KernelParameters(Base.size(λCC.data), λCC.data.offsets)
    launch!(CPU(), grid, kp, _compute_tripolar_coordinates!,
            λFC, φFC, λCC, φCC,
            λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶜᵃ,
            first_pole_longitude,
            focal_distance, Nx)
    kp = KernelParameters(Base.size(λCF.data), λCF.data.offsets)
    launch!(CPU(), grid, kp, _compute_tripolar_coordinates!,
            λFF, φFF, λCF, φCF,
            λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶠᵃ,
            first_pole_longitude,
            focal_distance, Nx)

    # Coordinates
    λᶠᶠᵃ = dropdims(λFF.data, dims=3)
    φᶠᶠᵃ = dropdims(φFF.data, dims=3)
    λᶠᶜᵃ = dropdims(λFC.data, dims=3)
    φᶠᶜᵃ = dropdims(φFC.data, dims=3)
    λᶜᶠᵃ = dropdims(λCF.data, dims=3)
    φᶜᶠᵃ = dropdims(φCF.data, dims=3)
    λᶜᶜᵃ = dropdims(λCC.data, dims=3)
    φᶜᶜᵃ = dropdims(φCC.data, dims=3)

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
    FF = Field{Face,   Face,   Center}(grid)
    FC = Field{Face,   Center, Center}(grid)
    CF = Field{Center, Face,   Center}(grid)
    CC = Field{Center, Center, Center}(grid)

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


    # Continue the metrics to the south with a LatitudeLongitudeGrid
    # metrics (probably we don't even need to do this, since the tripolar grid should
    # terminate below Antartica, but it's better to be safe)

    latitude_longitude_grid = LatitudeLongitudeGrid(; size,
                                                      latitude,
                                                      longitude,
                                                      halo,
                                                      z = (0, 1), # z does not really matter here
                                                      radius)

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
    grid = OrthogonalSphericalShellGrid{topology...}(arch,
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

    for j in Hy+1:0, i in Hx+1:Nx+Hx
        @inbounds new_metric[i, j] = lat_lon_metric[j]
    end

    return nothing
end

function Grids.with_halo(new_halo, old_grid::TripolarGrid)

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
                            southernmost_latitude,
                            fold_topology = topology(old_grid, 2))

    return new_grid
end
