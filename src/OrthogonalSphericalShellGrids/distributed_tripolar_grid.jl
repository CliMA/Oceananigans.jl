using MPI
using Oceananigans.BoundaryConditions: DistributedCommunicationBoundaryCondition
using Oceananigans.Fields: validate_indices, validate_field_data
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations:
    local_size,
    barrier!,
    all_reduce,
    ranks,
    inject_halo_communication_boundary_conditions,
    concatenate_local_sizes,
    communication_buffers

using Oceananigans.Grids: topology, RightConnected, FullyConnected

import Oceananigans.DistributedComputations: reconstruct_global_grid
import Oceananigans.Fields: Field, validate_indices, validate_boundary_conditions

const DistributedTripolarGrid{FT, TX, TY, TZ, CZ, CC, FC, CF, FF, Arch} =
    OrthogonalSphericalShellGrid{FT, TX, TY, TZ, CZ, <:Tripolar, CC, FC, CF, FF, <:Distributed{<:Union{CPU, GPU}}}

const DistributedTripolarGridOfSomeKind = Union{
    DistributedTripolarGrid,
    ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:DistributedTripolarGrid}
}

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
    px = ifelse(isnothing(arch.partition.x), 1, arch.partition.x)
    py = ifelse(isnothing(arch.partition.y), 1, arch.partition.y)

    # Check that partitioning in x is correct:
    try
        if isodd(px) && (px != 1)
            throw(ArgumentError("Only even partitioning in x is supported with the TripolarGrid"))
        end
    catch
        throw(ArgumentError("The x partition $(px) is not supported. The partition in x must be an even number. "))
    end

    # a slab decomposition in x is not supported
    if px != 1 && py == 1
        throw(ArgumentError("A x-only partitioning is not supported with the TripolarGrid. \n
                            Please, use a y partitioning configuration or a x-y pencil partitioning."))
    end

    Hx, Hy, Hz = halo

    # We build the global grid on a CPU architecture, in order to split it easily
    global_grid = TripolarGrid(CPU(), FT; halo, kwargs...)
    Nx, Ny, Nz = global_size = size(global_grid)

    # Splitting the grid manually
    lsize = local_size(arch, global_size)

    # Extracting the local range
    nxlocal = concatenate_local_sizes(lsize, arch, 1)
    nylocal = concatenate_local_sizes(lsize, arch, 2)
    xrank   = ifelse(isnothing(arch.partition.x), 0, arch.local_index[1] - 1)
    yrank   = ifelse(isnothing(arch.partition.y), 0, arch.local_index[2] - 1)

    # The j-range
    jstart = 1 + sum(nylocal[1:yrank])
    jend = yrank == workers[2] - 1 ? Ny : sum(nylocal[1:yrank+1])
    jrange = jstart-Hy:jend+Hy

    # The i-range
    istart = 1 + sum(nxlocal[1:xrank])
    iend = xrank == workers[1] - 1 ? Nx : sum(nxlocal[1:xrank+1])
    irange = istart-Hx:iend+Hx

    # Partitioning the Coordinates
    λᶠᶠᵃ = partition_tripolar_metric(global_grid, :λᶠᶠᵃ, irange, jrange)
    φᶠᶠᵃ = partition_tripolar_metric(global_grid, :φᶠᶠᵃ, irange, jrange)
    λᶠᶜᵃ = partition_tripolar_metric(global_grid, :λᶠᶜᵃ, irange, jrange)
    φᶠᶜᵃ = partition_tripolar_metric(global_grid, :φᶠᶜᵃ, irange, jrange)
    λᶜᶠᵃ = partition_tripolar_metric(global_grid, :λᶜᶠᵃ, irange, jrange)
    φᶜᶠᵃ = partition_tripolar_metric(global_grid, :φᶜᶠᵃ, irange, jrange)
    λᶜᶜᵃ = partition_tripolar_metric(global_grid, :λᶜᶜᵃ, irange, jrange)
    φᶜᶜᵃ = partition_tripolar_metric(global_grid, :φᶜᶜᵃ, irange, jrange)

    # Partitioning the Metrics
    Δxᶜᶜᵃ = partition_tripolar_metric(global_grid, :Δxᶜᶜᵃ, irange, jrange)
    Δxᶠᶜᵃ = partition_tripolar_metric(global_grid, :Δxᶠᶜᵃ, irange, jrange)
    Δxᶜᶠᵃ = partition_tripolar_metric(global_grid, :Δxᶜᶠᵃ, irange, jrange)
    Δxᶠᶠᵃ = partition_tripolar_metric(global_grid, :Δxᶠᶠᵃ, irange, jrange)
    Δyᶜᶜᵃ = partition_tripolar_metric(global_grid, :Δyᶜᶜᵃ, irange, jrange)
    Δyᶠᶜᵃ = partition_tripolar_metric(global_grid, :Δyᶠᶜᵃ, irange, jrange)
    Δyᶜᶠᵃ = partition_tripolar_metric(global_grid, :Δyᶜᶠᵃ, irange, jrange)
    Δyᶠᶠᵃ = partition_tripolar_metric(global_grid, :Δyᶠᶠᵃ, irange, jrange)
    Azᶜᶜᵃ = partition_tripolar_metric(global_grid, :Azᶜᶜᵃ, irange, jrange)
    Azᶠᶜᵃ = partition_tripolar_metric(global_grid, :Azᶠᶜᵃ, irange, jrange)
    Azᶜᶠᵃ = partition_tripolar_metric(global_grid, :Azᶜᶠᵃ, irange, jrange)
    Azᶠᶠᵃ = partition_tripolar_metric(global_grid, :Azᶠᶠᵃ, irange, jrange)

    LY = yrank == 0 ? RightConnected : FullyConnected
    LX = workers[1] == 1 ? Periodic : FullyConnected
    ny = nylocal[yrank+1]
    nx = nxlocal[xrank+1]

    z = on_architecture(arch, global_grid.z)
    radius = global_grid.radius

    # Fix corners halos passing in case workers[1] != 1
    if  workers[1] != 1
        northwest_idx_x = ranks(arch)[1] - arch.local_index[1] + 2
        northeast_idx_x = ranks(arch)[1] - arch.local_index[1]

        if northwest_idx_x > workers[1]
            northwest_idx_x = arch.local_index[1]
        end

        if northeast_idx_x < 1
            northeast_idx_x = arch.local_index[1]
        end

        # Make sure the northwest and northeast connectivities are correct
        northwest_recv_rank = receiving_rank(arch; receive_idx_x = northwest_idx_x)
        northeast_recv_rank = receiving_rank(arch; receive_idx_x = northeast_idx_x)
        north_recv_rank     = receiving_rank(arch)

        if yrank == workers[2] - 1
            arch.connectivity.northwest = northwest_recv_rank
            arch.connectivity.northeast = northeast_recv_rank
            arch.connectivity.north     = north_recv_rank
        end
    end

    FT   = eltype(global_grid)
    grid = OrthogonalSphericalShellGrid{LX, LY, Bounded}(arch,
                                                         nx, ny, Nz,
                                                         Hx, Hy, Hz,
                                                         convert(FT, global_grid.Lz),
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
                                                         global_grid.conformal_mapping)

    return grid
end

function partition_tripolar_metric(global_grid, metric_name, irange, jrange)

    metric = getproperty(global_grid, metric_name)
    offsets = metric.offsets
    partitioned_metric = metric[irange, jrange]

    if partitioned_metric isa OffsetArray
        partitioned_metric = partitioned_metric.parent
    end

    return OffsetArray(partitioned_metric, offsets...)
end

#####
##### Boundary condition extensions
#####

struct ZipperHaloCommunicationRanks{F, T, S}
    from :: F
      to :: T
    sign :: S
end

ZipperHaloCommunicationRanks(sign; from, to) = ZipperHaloCommunicationRanks(from, to, sign)

Base.summary(hcr::ZipperHaloCommunicationRanks) = "ZipperHaloCommunicationRanks from rank $(hcr.from) to rank $(hcr.to)"

# Finding out the paired rank to communicate the north boundary
# in case of a DistributedZipperBoundaryCondition using a "Handshake" procedure
function receiving_rank(arch; receive_idx_x = ranks(arch)[1] - arch.local_index[1] + 1)

    Ry = ranks(arch)[2]
    receive_rank  = 0

    for rank in 0:prod(ranks(arch)) - 1
        my_x_idx = 0
        my_y_idx = 0

        if arch.local_rank == rank
            my_x_idx = arch.local_index[1]
            my_y_idx = arch.local_index[2]
        end

        x_idx = all_reduce(+, my_x_idx, arch)
        y_idx = all_reduce(+, my_y_idx, arch)

        if x_idx == receive_idx_x && y_idx == Ry
            receive_rank = rank
        end
    end

    return receive_rank
end

# a distributed `TripolarGrid` needs a `ZipperBoundaryCondition` for the north boundary
# only on the last rank
function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              grid::DistributedTripolarGridOfSomeKind,
                                              field_name::Symbol,
                                              prognostic_names=nothing)

    arch = architecture(grid)
    loc  = assumed_field_location(field_name)
    yrank = arch.local_index[2] - 1

    processor_size = ranks(arch)
    sign = (field_name == :u) || (field_name == :v) ? -1 : 1

    west  = regularize_boundary_condition(bcs.west,  grid, loc, 1, LeftBoundary,  prognostic_names)
    east  = regularize_boundary_condition(bcs.east,  grid, loc, 1, RightBoundary, prognostic_names)
    south = regularize_boundary_condition(bcs.south, grid, loc, 2, LeftBoundary,  prognostic_names)

    north = if yrank == processor_size[2] - 1 && processor_size[1] == 1
        ZipperBoundaryCondition(sign)

    elseif yrank == processor_size[2] - 1 && processor_size[1] != 1
        from = arch.local_rank
        # Search the rank to send to
        to = arch.connectivity.north
        halo_communication = ZipperHaloCommunicationRanks(sign; from, to)
        DistributedCommunicationBoundaryCondition(halo_communication)

    else
        regularize_boundary_condition(bcs.north, grid, loc, 2, RightBoundary, prognostic_names)

    end

    bottom = regularize_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top =    regularize_boundary_condition(bcs.top,    grid, loc, 3, RightBoundary, prognostic_names)

    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

# Extension of the constructor for a `Field` on a `TRG` grid. We assumes that the north boundary is a zipper
# with a sign that depends on the location of the field (revert the value of the halos if on edges, keep it if on nodes or centers)
function Field((LX, LY, LZ)::Tuple, grid::DistributedTripolarGridOfSomeKind, data, old_bcs, indices::Tuple, op, status)
    arch = architecture(grid)
    yrank = arch.local_index[2] - 1

    processor_size = ranks(arch)

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
        if yrank == processor_size[2] - 1 && processor_size[1] == 1
            north_bc = if !(old_bcs.north isa ZBC)
                default_zipper
            else
                old_bcs.north
            end

        elseif yrank == processor_size[2] - 1 && processor_size[1] != 1
            sgn  = old_bcs.north isa ZBC ? old_bcs.north.condition : sign(LX, LY)
            from = arch.local_rank
            to   = arch.connectivity.north
            halo_communication = ZipperHaloCommunicationRanks(sgn; from, to)
            north_bc = DistributedCommunicationBoundaryCondition(halo_communication)

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

    buffers = communication_buffers(grid, data, new_bcs)

    return Field{LX, LY, LZ}(grid, data, new_bcs, indices, op, status, buffers)
end

# Reconstruction the global tripolar grid for visualization purposes
function reconstruct_global_grid(grid::DistributedTripolarGrid)

    arch = grid.architecture

    n    = Base.size(grid)
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

function with_halo(new_halo, old_grid::DistributedTripolarGrid)

    arch = old_grid.architecture

    n  = size(old_grid)
    N  = map(sum, concatenate_local_sizes(n, arch))
    z  = cpu_face_constructor_z(old_grid)

    north_poles_latitude = old_grid.conformal_mapping.north_poles_latitude
    first_pole_longitude = old_grid.conformal_mapping.first_pole_longitude
    southernmost_latitude = old_grid.conformal_mapping.southernmost_latitude

    return TripolarGrid(arch, eltype(old_grid);
                        halo = new_halo,
                        size = N,
                        north_poles_latitude,
                        first_pole_longitude,
                        southernmost_latitude,
                        z)
end
