using Oceananigans.BoundaryConditions: UPivotZipperBoundaryCondition, FPivotZipperBoundaryCondition, NoFluxBoundaryCondition
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
const TripolarGridOfSomeKind{FT, TX, TY, TZ} = Union{TripolarGrid{FT, TX, TY, TZ}, ImmersedBoundaryGrid{FT, TX, TY, TZ, <:TripolarGrid}}

"""
    TripolarGrid(arch = CPU(), FT::DataType = Oceananigans.defaults.FloatType;
                 size,
                 southernmost_latitude = -80,
                 halo = (4, 4, 4),
                 radius = Oceananigans.defaults.planet_radius,
                 z = (0, 1),
                 north_poles_latitude = 55,
                 first_pole_longitude = 70,
                 fold_topology = RightCenterFolded)

Return an `OrthogonalSphericalShellGrid` tripolar grid on the sphere. The
tripolar grid replaces the North Pole singularity with two other singularities
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

!!! warning "Longitude coordinate must have an even number of cells"
    `size` is a 3-tuple of the grid size in longitude, latitude, and vertical directions.
    Due to requirements of the folding at the north edge of the domain, the longitude size
    of the grid (i.e., the first component of `size`) _must_ be an even number!

!!! info "North pole singularities"
    For a `RightCenterFolded` y-topology, The north singularities are located on `(Face, Center)`,
    at: `i = 1`, `j = grid.Ny` and `i = grid.Nx ÷ 2 + 1`, `j = grid.Ny`.
    Pivot points are indicated by the `↻` symbols below:
    ```
              │           │           │           │           │           │           │
    Ny+1 ─▶ ──╔═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╗──
              ║           │           │           │           │           │           ║
    Ny   ─▶   ↻     c     u     c     u     c     ↻     c     u     c     u     c     ↻ ◀─ Fold
              ║           │           │           │           │           │           ║
    Ny   ─▶ ──╫──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────╫──
              ║           │           │           │           │           │           ║
    Ny-1 ─▶   u     c     u     c     u     c     u     c     u     c     u     c     ║
              ║           │           │           │           │           │           ║
    Ny-1 ─▶ ──╫──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────╫──
              ║           │           │           │           │           │           ║
              ▲     ▲     ▲                       ▲                       ▲     ▲     ▲
              1     1     2                     Nx÷2+1                    Nx    Nx    Nx+1
    ```
    See [`UPivotZipperBoundaryCondition`](@ref) for more information on the fold.

    For a `RightFaceFolded` y-topology, the fold is located along the y-direction faces
    at `j = Ny+1` (i.e., the fold is exactly on the northern boundary of the grid).
    The north singularities are located on `(Face, Face)`.

    The fold is located between the last interior face row and the first halo face row.
    Pivot points (↻) are located on `(Face, Face)`:
    ```
              │           │           │           │           │           │           │
    Ny+1 ─▶ ─ ↻ ═══ v ════╪════ v ════╪════ v ═══ ↻ ═══ v ════╪════ v ════╪════ v ═══ ↻ ◀─ Fold
              ║           │           │           │           │           │           ║ (at yface[Ny+1])
    Ny   ─▶   u     c     u     c     u     c     u     c     u     c     u     c     ║
              ║           │           │           │           │           │           ║
    Ny   ─▶ ──╫──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────╫──
              ║           │           │           │           │           │           ║
    Ny-1 ─▶   u     c     u     c     u     c     u     c     u     c     u     c     ║
              ║           │           │           │           │           │           ║
    Ny-1 ─▶ ──╫──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────┼──── v ────╫──
              ║           │           │           │           │           │           ║
              ▲     ▲     ▲                       ▲                       ▲     ▲     ▲
              1     1     2                     Nx÷2+1                    Nx    Nx    Nx+1
    ```
    See [`FPivotZipperBoundaryCondition`](@ref) for more information on the fold.


References
==========

Murray, R. J. (1996). Explicit generation of orthogonal grids for ocean models.
    Journal of Computational Physics, 126(2), 251-273.
"""
function TripolarGrid(arch = CPU(), FT::DataType = Oceananigans.defaults.FloatType;
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

    # Make sure φ's are valid in the south
    if φᵃᶠᵃ[1] < -90
        msg = "Your southernmost latitude is too far South! (The southernmost grid cell does not fit.)"
        if topology[2] === RightCenterFolded
            msg *= '\n' * "For latitude with $(topology[2]) topology, you need to ensure that southernmost_latitude - Δφ/2 ≥ -90."
        end
        throw(ArgumentError(msg))
    end

    # return λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC
    # Helper grid to launch kernels on
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
    kp = KernelParameters(1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
    launch!(CPU(), grid, kp, _compute_tripolar_coordinates!,
        λFC, φFC, λCC, φCC,
        λFF, φFF, λCF, φCF,
        λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶜᵃ, φᵃᶠᵃ,
        first_pole_longitude,
        focal_distance, Nx, Ny
    )

    # Coordinates
    λᶠᶠᵃ = dropdims(λFF.data, dims=3)
    φᶠᶠᵃ = dropdims(φFF.data, dims=3)
    λᶠᶜᵃ = dropdims(λFC.data, dims=3)
    φᶠᶜᵃ = dropdims(φFC.data, dims=3)
    λᶜᶠᵃ = dropdims(λCF.data, dims=3)
    φᶜᶠᵃ = dropdims(φCF.data, dims=3)
    λᶜᶜᵃ = dropdims(λCC.data, dims=3)
    φᶜᶜᵃ = dropdims(φCC.data, dims=3)

    # Boundary conditions to fill halos of the metric terms
    # We define them manually because the helper RectilinearGrid
    # does not know how to fold the north boundary...
    boundary_conditions = FieldBoundaryConditions(north  = north_fold_boundary_condition(fold_topology)(),
                                                  south  = NoFluxBoundaryCondition(), # The south should be `continued`
                                                  west   = Oceananigans.PeriodicBoundaryCondition(),
                                                  east   = Oceananigans.PeriodicBoundaryCondition(),
                                                  top    = nothing,
                                                  bottom = nothing)


    # return metrics grid
    grid = RectilinearGrid(; size = (Nx, Ny),
                             halo = (Hx, Hy),
                             x = (0, 1), y = (0, 1),
                             topology = (Periodic, fold_topology, Flat))

    # Allocate fields for all the metrics fields to fill halos
    # TODO: make these on_architecture(arch, zeros(Nx, Ny))
    # to build the grid on GPU
    Δxᶠᶠᵃ = Field{Face,   Face,   Center}(grid; boundary_conditions)
    Δxᶠᶜᵃ = Field{Face,   Center, Center}(grid; boundary_conditions)
    Δxᶜᶠᵃ = Field{Center, Face,   Center}(grid; boundary_conditions)
    Δxᶜᶜᵃ = Field{Center, Center, Center}(grid; boundary_conditions)
    Δyᶠᶠᵃ = Field{Face,   Face,   Center}(grid; boundary_conditions)
    Δyᶠᶜᵃ = Field{Face,   Center, Center}(grid; boundary_conditions)
    Δyᶜᶠᵃ = Field{Center, Face,   Center}(grid; boundary_conditions)
    Δyᶜᶜᵃ = Field{Center, Center, Center}(grid; boundary_conditions)
    Azᶠᶠᵃ = Field{Face,   Face,   Center}(grid; boundary_conditions)
    Azᶠᶜᵃ = Field{Face,   Center, Center}(grid; boundary_conditions)
    Azᶜᶠᵃ = Field{Center, Face,   Center}(grid; boundary_conditions)
    Azᶜᶜᵃ = Field{Center, Center, Center}(grid; boundary_conditions)

    # Calculate metrics
    # TODO: rewrite this kernel and split the call to match the indices exactly.
    kp = KernelParameters(1:Nx, 1:Ny+1)
    launch!(CPU(), grid, kp, _calculate_metrics!,
        Δxᶠᶜᵃ, Δxᶜᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
        Δyᶠᶜᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
        Azᶠᶜᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
        λᶠᶜᵃ, λᶜᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
        φᶠᶜᵃ, φᶜᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ,
        radius
    )

    # Fill all halos
    fill_halo_regions!(Δxᶠᶠᵃ)
    fill_halo_regions!(Δxᶠᶜᵃ)
    fill_halo_regions!(Δxᶜᶠᵃ)
    fill_halo_regions!(Δxᶜᶜᵃ)
    fill_halo_regions!(Δyᶠᶠᵃ)
    fill_halo_regions!(Δyᶠᶜᵃ)
    fill_halo_regions!(Δyᶜᶠᵃ)
    fill_halo_regions!(Δyᶜᶜᵃ)
    fill_halo_regions!(Azᶠᶠᵃ)
    fill_halo_regions!(Azᶠᶜᵃ)
    fill_halo_regions!(Azᶜᶠᵃ)
    fill_halo_regions!(Azᶜᶜᵃ)

    # Copy metrics as offset arrays
    Δxᶠᶠᵃ = deepcopy(dropdims(Δxᶠᶠᵃ.data, dims=3))
    Δxᶜᶠᵃ = deepcopy(dropdims(Δxᶜᶠᵃ.data, dims=3))
    Δxᶠᶜᵃ = deepcopy(dropdims(Δxᶠᶜᵃ.data, dims=3))
    Δxᶜᶜᵃ = deepcopy(dropdims(Δxᶜᶜᵃ.data, dims=3))
    Δyᶠᶠᵃ = deepcopy(dropdims(Δyᶠᶠᵃ.data, dims=3))
    Δyᶜᶠᵃ = deepcopy(dropdims(Δyᶜᶠᵃ.data, dims=3))
    Δyᶠᶜᵃ = deepcopy(dropdims(Δyᶠᶜᵃ.data, dims=3))
    Δyᶜᶜᵃ = deepcopy(dropdims(Δyᶜᶜᵃ.data, dims=3))
    Azᶠᶠᵃ = deepcopy(dropdims(Azᶠᶠᵃ.data, dims=3))
    Azᶜᶠᵃ = deepcopy(dropdims(Azᶜᶠᵃ.data, dims=3))
    Azᶠᶜᵃ = deepcopy(dropdims(Azᶠᶜᵃ.data, dims=3))
    Azᶜᶜᵃ = deepcopy(dropdims(Azᶜᶜᵃ.data, dims=3))

    # Continue the metrics to the south with a LatitudeLongitudeGrid
    # metrics (probably we don't even need to do this, since the tripolar grid should
    # terminate below Antarctica, but it's better to be safe)
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

    for j in Hy+1:1, i in Hx+1:Nx+Hx
        @inbounds new_metric[i, j] = lat_lon_metric[j]
    end

    return nothing
end

# Copy interior data from old grid into a new Field, then fill halos
# using the proper boundary conditions (periodic in x, fold at north, no-flux at south).
function transfer_horizontal_field(old_data, helper_grid, bcs, LX, LY)
    TX, TY, _ = topology(helper_grid)
    Nx, Ny, _ = size(helper_grid)
    new_field = Field{LX, LY, Center}(helper_grid; boundary_conditions = bcs)
    Ni = Base.length(LX(), TX(), Nx)
    Nj = Base.length(LY(), TY(), Ny)
    cpu_old_data = on_architecture(CPU(), old_data)
    new_field.data[1:Ni, 1:Nj, 1] .= cpu_old_data[1:Ni, 1:Nj]
    fill_halo_regions!(new_field)
    return deepcopy(dropdims(new_field.data, dims=3))
end

function Grids.with_halo(new_halo, old_grid::TripolarGrid)

    arch = architecture(old_grid)
    FT = eltype(old_grid)

    Nx,  Ny,  Nz  = size(old_grid)
    TX,  TY,  TZ  = topology(old_grid)
    Hxo, Hyo, Hzo = halo_size(old_grid)
    Hxn, Hyn, Hzn = new_halo

    # Reconstruct vertical coordinate with new halo
    z = cpu_face_constructor_z(old_grid)
    Lz, new_z = generate_coordinate(FT, topology(old_grid), (Nx, Ny, Nz), new_halo, z, :z, 3, CPU())

    # Helper grid for halo filling (same approach as the TripolarGrid constructor)
    helper_grid = RectilinearGrid(; size = (Nx, Ny),
                                    halo = (Hxn, Hyn),
                                    x = (0, 1), y = (0, 1),
                                    topology = (TX, TY, Flat))

    # Boundary conditions for halo filling (same as in the TripolarGrid constructor)
    bcs = FieldBoundaryConditions(north  = north_fold_boundary_condition(TY)(),
                                  south  = NoFluxBoundaryCondition(),
                                  west   = Oceananigans.PeriodicBoundaryCondition(),
                                  east   = Oceananigans.PeriodicBoundaryCondition(),
                                  top    = nothing,
                                  bottom = nothing)

    λᶜᶜᵃ  = transfer_horizontal_field(old_grid.λᶜᶜᵃ,  helper_grid, bcs, Center, Center)
    λᶠᶜᵃ  = transfer_horizontal_field(old_grid.λᶠᶜᵃ,  helper_grid, bcs, Face,   Center)
    λᶜᶠᵃ  = transfer_horizontal_field(old_grid.λᶜᶠᵃ,  helper_grid, bcs, Center, Face)
    λᶠᶠᵃ  = transfer_horizontal_field(old_grid.λᶠᶠᵃ,  helper_grid, bcs, Face,   Face)

    φᶜᶜᵃ  = transfer_horizontal_field(old_grid.φᶜᶜᵃ,  helper_grid, bcs, Center, Center)
    φᶠᶜᵃ  = transfer_horizontal_field(old_grid.φᶠᶜᵃ,  helper_grid, bcs, Face,   Center)
    φᶜᶠᵃ  = transfer_horizontal_field(old_grid.φᶜᶠᵃ,  helper_grid, bcs, Center, Face)
    φᶠᶠᵃ  = transfer_horizontal_field(old_grid.φᶠᶠᵃ,  helper_grid, bcs, Face,   Face)

    Δxᶜᶜᵃ = transfer_horizontal_field(old_grid.Δxᶜᶜᵃ, helper_grid, bcs, Center, Center)
    Δxᶠᶜᵃ = transfer_horizontal_field(old_grid.Δxᶠᶜᵃ, helper_grid, bcs, Face,   Center)
    Δxᶜᶠᵃ = transfer_horizontal_field(old_grid.Δxᶜᶠᵃ, helper_grid, bcs, Center, Face)
    Δxᶠᶠᵃ = transfer_horizontal_field(old_grid.Δxᶠᶠᵃ, helper_grid, bcs, Face,   Face)

    Δyᶜᶜᵃ = transfer_horizontal_field(old_grid.Δyᶜᶜᵃ, helper_grid, bcs, Center, Center)
    Δyᶠᶜᵃ = transfer_horizontal_field(old_grid.Δyᶠᶜᵃ, helper_grid, bcs, Face,   Center)
    Δyᶜᶠᵃ = transfer_horizontal_field(old_grid.Δyᶜᶠᵃ, helper_grid, bcs, Center, Face)
    Δyᶠᶠᵃ = transfer_horizontal_field(old_grid.Δyᶠᶠᵃ, helper_grid, bcs, Face,   Face)

    Azᶜᶜᵃ = transfer_horizontal_field(old_grid.Azᶜᶜᵃ, helper_grid, bcs, Center, Center)
    Azᶠᶜᵃ = transfer_horizontal_field(old_grid.Azᶠᶜᵃ, helper_grid, bcs, Face,   Center)
    Azᶜᶠᵃ = transfer_horizontal_field(old_grid.Azᶜᶠᵃ, helper_grid, bcs, Center, Face)
    Azᶠᶠᵃ = transfer_horizontal_field(old_grid.Azᶠᶠᵃ, helper_grid, bcs, Face,   Face)

    new_grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(arch,
                                                         Nx, Ny, Nz,
                                                         Hxn, Hyn, Hzn,
                                                         convert(FT, Lz),
                                                         on_architecture(arch, λᶜᶜᵃ),
                                                         on_architecture(arch, λᶠᶜᵃ),
                                                         on_architecture(arch, λᶜᶠᵃ),
                                                         on_architecture(arch, λᶠᶠᵃ),
                                                         on_architecture(arch, φᶜᶜᵃ),
                                                         on_architecture(arch, φᶠᶜᵃ),
                                                         on_architecture(arch, φᶜᶠᵃ),
                                                         on_architecture(arch, φᶠᶠᵃ),
                                                         on_architecture(arch, new_z),
                                                         on_architecture(arch, Δxᶜᶜᵃ),
                                                         on_architecture(arch, Δxᶠᶜᵃ),
                                                         on_architecture(arch, Δxᶜᶠᵃ),
                                                         on_architecture(arch, Δxᶠᶠᵃ),
                                                         on_architecture(arch, Δyᶜᶜᵃ),
                                                         on_architecture(arch, Δyᶠᶜᵃ),
                                                         on_architecture(arch, Δyᶜᶠᵃ),
                                                         on_architecture(arch, Δyᶠᶠᵃ),
                                                         on_architecture(arch, Azᶜᶜᵃ),
                                                         on_architecture(arch, Azᶠᶜᵃ),
                                                         on_architecture(arch, Azᶜᶠᵃ),
                                                         on_architecture(arch, Azᶠᶠᵃ),
                                                         convert(FT, old_grid.radius),
                                                         old_grid.conformal_mapping)

    return new_grid
end
