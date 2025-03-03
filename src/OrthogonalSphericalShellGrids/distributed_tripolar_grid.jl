using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: local_size,
                                            barrier!,
                                            ranks,
                                            inject_halo_communication_boundary_conditions,
                                            concatenate_local_sizes

using Oceananigans.Grids: topology

import Oceananigans.DistributedComputations: reconstruct_global_grid

const DistributedTripolarGrid{FT, TX, TY, TZ, CZ, A, Arch} =
    OrthogonalSphericalShellGrid{FT, TX, TY, TZ, CZ, A, <:Tripolar, <:Distributed}

const DTRG = Union{DistributedTripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:DistributedTripolarGrid}}

"""
    TripolarGrid(arch::Distributed, FT::DataType = Float64; halo = (4, 4, 4), kwargs...)

Construct a tripolar grid on a distributed architecture.
A distributed tripolar grid is supported only on a Y-partitioning configuration,
therefore, only splitting the j-direction is supported for the moment.
"""
function TripolarGrid(arch::Distributed, FT::DataType=Float64;
                      halo=(4, 4, 4),
                      kwargs...)

    workers = ranks(arch.partition)

    workers[1] != 1 &&
        throw(ArgumentError("The tripolar grid is supported only on a Y-partitioning configuration"))

    Hx, Hy, Hz = halo

    # We build the global grid on a CPU architecture, in order to split it easily
    global_grid = TripolarGrid(CPU(), FT; halo, kwargs...)
    Nx, Ny, Nz = global_size = size(global_grid)

    # Splitting the grid manually, remember, only splitting
    # the j-direction is supported for the moment
    lsize = local_size(arch, global_size)

    # Extracting the local range
    nlocal = concatenate_local_sizes(lsize, arch, 2)
    rank = arch.local_rank

    jstart = 1 + sum(nlocal[1:rank])
    jend = rank == workers[2] - 1 ? Ny : sum(nlocal[1:rank+1])
    jrange = jstart-Hy:jend+Hy

    # Partitioning the Coordinates
    λᶠᶠᵃ = partition_tripolar_metric(global_grid, :λᶠᶠᵃ, jrange)
    φᶠᶠᵃ = partition_tripolar_metric(global_grid, :φᶠᶠᵃ, jrange)
    λᶠᶜᵃ = partition_tripolar_metric(global_grid, :λᶠᶜᵃ, jrange)
    φᶠᶜᵃ = partition_tripolar_metric(global_grid, :φᶠᶜᵃ, jrange)
    λᶜᶠᵃ = partition_tripolar_metric(global_grid, :λᶜᶠᵃ, jrange)
    φᶜᶠᵃ = partition_tripolar_metric(global_grid, :φᶜᶠᵃ, jrange)
    λᶜᶜᵃ = partition_tripolar_metric(global_grid, :λᶜᶜᵃ, jrange)
    φᶜᶜᵃ = partition_tripolar_metric(global_grid, :φᶜᶜᵃ, jrange)

    # Partitioning the Metrics
    Δxᶜᶜᵃ = partition_tripolar_metric(global_grid, :Δxᶜᶜᵃ, jrange)
    Δxᶠᶜᵃ = partition_tripolar_metric(global_grid, :Δxᶠᶜᵃ, jrange)
    Δxᶜᶠᵃ = partition_tripolar_metric(global_grid, :Δxᶜᶠᵃ, jrange)
    Δxᶠᶠᵃ = partition_tripolar_metric(global_grid, :Δxᶠᶠᵃ, jrange)
    Δyᶜᶜᵃ = partition_tripolar_metric(global_grid, :Δyᶜᶜᵃ, jrange)
    Δyᶠᶜᵃ = partition_tripolar_metric(global_grid, :Δyᶠᶜᵃ, jrange)
    Δyᶜᶠᵃ = partition_tripolar_metric(global_grid, :Δyᶜᶠᵃ, jrange)
    Δyᶠᶠᵃ = partition_tripolar_metric(global_grid, :Δyᶠᶠᵃ, jrange)
    Azᶜᶜᵃ = partition_tripolar_metric(global_grid, :Azᶜᶜᵃ, jrange)
    Azᶠᶜᵃ = partition_tripolar_metric(global_grid, :Azᶠᶜᵃ, jrange)
    Azᶜᶠᵃ = partition_tripolar_metric(global_grid, :Azᶜᶠᵃ, jrange)
    Azᶠᶠᵃ = partition_tripolar_metric(global_grid, :Azᶠᶠᵃ, jrange)

    LY = rank == 0 ? RightConnected : FullyConnected
    ny = nlocal[rank+1]

    z = global_grid.z
    radius = global_grid.radius

    grid = OrthogonalSphericalShellGrid{Periodic, LY, Bounded}(arch,
                                                               Nx, ny, Nz,
                                                               Hx, Hy, Hz,
                                                               convert(eltype(radius), global_grid.Lz),
                                                               on_architecture(arch, λᶜᶜᵃ),
                                                               on_architecture(arch, λᶠᶜᵃ),
                                                               on_architecture(arch, λᶜᶠᵃ),
                                                               on_architecture(arch, λᶠᶠᵃ),
                                                               on_architecture(arch, φᶜᶜᵃ),
                                                               on_architecture(arch, φᶠᶜᵃ),
                                                               on_architecture(arch, φᶜᶠᵃ),
                                                               on_architecture(arch, φᶠᶠᵃ),
                                                               on_architecture(arch, z),
                                                               on_architecture(arch, Δxᶜᶜᵃ),
                                                               on_architecture(arch, Δxᶠᶜᵃ),
                                                               on_architecture(arch, Δxᶜᶠᵃ),
                                                               on_architecture(arch, Δxᶠᶠᵃ),
                                                               on_architecture(arch, Δyᶜᶜᵃ),
                                                               on_architecture(arch, Δyᶜᶠᵃ),
                                                               on_architecture(arch, Δyᶠᶜᵃ),
                                                               on_architecture(arch, Δyᶠᶠᵃ),
                                                               on_architecture(arch, Azᶜᶜᵃ),
                                                               on_architecture(arch, Azᶠᶜᵃ),
                                                               on_architecture(arch, Azᶜᶠᵃ),
                                                               on_architecture(arch, Azᶠᶠᵃ),
                                                               radius,
                                                               global_grid.conformal_mapping)

    return grid
end

function partition_tripolar_metric(global_grid, metric_name, jrange)

    metric = getproperty(global_grid, metric_name)
    offsets = metric.offsets

    partitioned_metric = metric[:, jrange].parent

    return OffsetArray(partitioned_metric, offsets...)
end


#####
##### Boundary condition extensions
#####

# a distributed `TripolarGrid` needs a `ZipperBoundaryCondition` for the north boundary
# only on the last rank
function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
    grid::DTRG,
    field_name::Symbol,
    prognostic_names=nothing)

    arch = architecture(grid)
    loc  = assumed_field_location(field_name)
    rank = arch.local_rank
    processor_size = ranks(arch.partition)
    sign = (field_name == :u) || (field_name == :v) ? -1 : 1

    west =  regularize_boundary_condition(bcs.west,  grid, loc, 1, LeftBoundary,  prognostic_names)
    east =  regularize_boundary_condition(bcs.east,  grid, loc, 1, RightBoundary, prognostic_names)
    south = regularize_boundary_condition(bcs.south, grid, loc, 2, LeftBoundary,  prognostic_names)
    north = if rank == processor_size[2] - 1
        ZipperBoundaryCondition(sign)
    else
        regularize_boundary_condition(bcs.south, grid, loc, 2, RightBoundary, prognostic_names)
    end

    bottom = regularize_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top =    regularize_boundary_condition(bcs.top,    grid, loc, 3, RightBoundary, prognostic_names)

    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

# Extension of the constructor for a `Field` on a `TRG` grid. We assumes that the north boundary is a zipper
# with a sign that depends on the location of the field (revert the value of the halos if on edges, keep it if on nodes or centers)
function Field((LX, LY, LZ)::Tuple, grid::DTRG, data, old_bcs, indices::Tuple, op, status)
    arch = architecture(grid)
    rank = arch.local_rank
    processor_size = ranks(arch.partition)
    indices = validate_indices(indices, (LX, LY, LZ), grid)
    validate_field_data((LX, LY, LZ), data, grid, indices)
    validate_boundary_conditions((LX, LY, LZ), grid, old_bcs)
    default_zipper = ZipperBoundaryCondition(sign(LX, LY))

    if isnothing(old_bcs) || ismissing(old_bcs)
        new_bcs = old_bcs
    else
        new_bcs = inject_halo_communication_boundary_conditions(old_bcs, arch.local_rank, arch.connectivity, topology(grid))

        # North boundary conditions are "special". If we are at the top of the domain, i.e.
        # the last rank, then we need to substitute the BC only if the old one is not already
        # a zipper boundary condition. Otherwise we always substitute because we need to 
        # inject the halo boundary conditions.
        if rank == processor_size[2] - 1
            north_bc = if !(old_bcs.north isa ZBC)
                default_zipper
            else
                old_bcs.north
            end
        else
            north_bc = new_bcs.north
        end

        new_bcs = FieldBoundaryConditions(; west=new_bcs.west,
            east=new_bcs.east,
            south=new_bcs.south,
            north=north_bc,
            top=new_bcs.top,
            bottom=new_bcs.bottom)
    end

    buffers = FieldBoundaryBuffers(grid, data, new_bcs)

    return Field{LX,LY,LZ}(grid, data, new_bcs, indices, op, status, buffers)
end

# Reconstruction the global tripolar grid for visualization purposes
function reconstruct_global_grid(grid::DistributedTripolarGrid)

    arch = grid.architecture

    n = size(grid)
    halo = halo_size(grid)
    size = map(sum, concatenate_local_sizes(n, arch))

    z = cpu_face_constructor_z(grid)

    child_arch = child_architecture(arch)

    FT = eltype(grid)

    north_poles_latitude = grid.conformal_mapping.north_poles_latitude
    first_pole_longitude = grid.conformal_mapping.first_pole_longitude
    southernmost_latitude = grid.conformal_mapping.southernmost_latitude

    return TripolarGrid(child_arch, FT;
        halo,
        size,
        north_poles_latitude,
        first_pole_longitude,
        southernmost_latitude,
        z)
end
