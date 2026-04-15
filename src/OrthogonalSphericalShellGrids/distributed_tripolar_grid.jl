using Oceananigans.BoundaryConditions: DistributedCommunicationBoundaryCondition, ZBC, DCBC
using Oceananigans.Fields: validate_indices, validate_field_data
using Oceananigans.DistributedComputations:
    DistributedComputations,
    Distributed,
    local_size,
    all_reduce,
    child_architecture,
    ranks,
    inject_halo_communication_boundary_conditions,
    concatenate_local_sizes,
    communication_buffers

using Oceananigans.Grids: topology, FullyConnected,
    RightCenterFolded, RightFaceFolded,
    LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded,
    LeftConnectedRightCenterConnected, LeftConnectedRightFaceConnected
using Oceananigans.DistributedComputations: insert_connected_topology
using Oceananigans.Utils: Utils

import Oceananigans.Fields: Field, validate_indices, validate_boundary_conditions
import Oceananigans.DistributedComputations: inject_halo_communication_boundary_conditions

const DistributedTripolarGrid{FT, TX, TY, TZ, CZ, CC, FC, CF, FF, Arch} =
    OrthogonalSphericalShellGrid{FT, TX, TY, TZ, CZ, <:Tripolar, CC, FC, CF, FF, <:Distributed{<:Union{CPU, GPU}}}

const MPITripolarGrid{FT, TX, TY, TZ, CZ, CC, FC, CF, FF, Arch} = OrthogonalSphericalShellGrid{FT, TX, TY, TZ, CZ, <:Tripolar, CC, FC, CF, FF, <:Distributed{<:Union{CPU, GPU}}}
const MPITripolarGridOfSomeKind = Union{MPITripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:MPITripolarGrid}}

# Defined here (rather than distributed_zipper.jl) because the include order
# requires distributed_tripolar_grid.jl before distributed_zipper.jl.
const OneDFoldTopology = Union{RightCenterFolded, RightFaceFolded,
                               LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded}

const TwoDFoldTopology = Union{LeftConnectedRightCenterConnected,
                               LeftConnectedRightFaceConnected}

"""
    TripolarGrid(arch::Distributed, FT::DataType = Float64; halo = (4, 4, 4), kwargs...)

Construct a tripolar grid on a distributed `arch`itecture.

!!! compat "Supported partitionings"

    Allowed partitionings include:
    - Only partition in `y`, e.g., `Distributed(CPU(), partition=Partition(1, 4))`.
    - Partition both in `x` and `y` with `x` partition even. For example:
      - `Distributed(CPU(), partition=Partition(2, 4))` is supported
      - `Distributed(CPU(), partition=Partition(3, 4))` is _not_ supported

    Note that partitioning only in `x`, e.g., `Distributed(CPU(), partition=Partition(4))`
    or `Distributed(CPU(), partition=Partition(4, 1))` is _not_ supported.
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
            throw(ArgumentError("Only even partitioning in x is supported with TripolarGrid."))
        end
    catch
        throw(ArgumentError("The x partition $(px) is not supported. The partition in x must be an even number."))
    end

    # a slab decomposition in x is not supported
    if px != 1 && py == 1
        throw(ArgumentError("An x-only partitioning is not supported for TripolarGrid. \n
                             Please, use a y partitioning configuration or an x-y pencil partitioning."))
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

    LX = workers[1] == 1 ? Periodic : FullyConnected

    global_fold_topology = topology(global_grid, 2)

    # 1-based indices for insert_connected_topology
    Rx, Ry = workers[1], workers[2]
    rx, ry = xrank + 1, yrank + 1
    LY = insert_connected_topology(global_fold_topology, Ry, ry, Rx, rx)
    ny = nylocal[ry]
    nx = nxlocal[rx]

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

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              grid::MPITripolarGridOfSomeKind,
                                              field_name::Symbol,
                                              prognostic_names=nothing)

    validate_boundary_condition_topology(bcs.north, topology(grid, 2)(), :north)

    arch = architecture(grid)
    loc  = assumed_field_location(field_name)
    yrank = arch.local_index[2] - 1

    processor_size = ranks(arch)
    sign = (field_name == :u) || (field_name == :v) ? -1 : 1

    west  = regularize_boundary_condition(bcs.west,  grid, loc, 1, LeftBoundary,  prognostic_names)
    east  = regularize_boundary_condition(bcs.east,  grid, loc, 1, RightBoundary, prognostic_names)
    south = regularize_boundary_condition(bcs.south, grid, loc, 2, LeftBoundary,  prognostic_names)

    north = if yrank == processor_size[2] - 1 && processor_size[1] == 1
        TY = fold_topology(grid.conformal_mapping)
        north_fold_boundary_condition(TY)(sign)

    elseif yrank == processor_size[2] - 1 && processor_size[1] != 1
        from = arch.local_rank
        to   = arch.connectivity.north
        halo_communication = ZipperHaloCommunicationRanks(sign; from, to)
        DistributedCommunicationBoundaryCondition(halo_communication)

    else
        regularize_boundary_condition(bcs.north, grid, loc, 2, RightBoundary, prognostic_names)

    end

    bottom   = regularize_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top      = regularize_boundary_condition(bcs.top,    grid, loc, 3, RightBoundary, prognostic_names)
    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

#####
##### Dispatch on (y-topology, north_bc) to determine the north zipper BC.
#####

# Extract the sign carried by the incoming north BC. Both dispatch levels below
# can receive either a `ZBC` (from default_auxiliary_bc or from the slab regularize
# branch) or a `DCBC` (from the pencil regularize branch, where the sign is packed
# inside a `ZipperHaloCommunicationRanks` alongside the MPI rank info).
zipper_sign(bc::ZBC)  = bc.condition
zipper_sign(bc::DCBC) = bc.condition.sign

# Non-fold topologies for non-fold ranks: no override
north_zipper_bc(topo, north_bc, loc, grid) = nothing

# Reduced fields have nothing north BCs (from default_auxiliary_bc) — no override
north_zipper_bc(::OneDFoldTopology, ::Nothing, loc, grid) = nothing
north_zipper_bc(::TwoDFoldTopology, ::Nothing, loc, grid) = nothing

##### Slab fold (1xN) or fold-north rank in pencil partition: local Zipper BC
function north_zipper_bc(::TY, north_bc, loc, grid) where TY <: OneDFoldTopology
    return north_fold_boundary_condition(TY)(zipper_sign(north_bc))
end

##### Middle rank in pencil partition (north side is MPI neighbor, not a fold):
##### wrap the sign into a `DistributedZipper` communication BC
function north_zipper_bc(::TwoDFoldTopology, north_bc, loc, grid)
    arch = architecture(grid)
    halo_communication = ZipperHaloCommunicationRanks(zipper_sign(north_bc); from=arch.local_rank, to=arch.connectivity.north)
    return DistributedCommunicationBoundaryCondition(halo_communication)
end

#####
##### Tripolar inject_halo_communication_boundary_conditions
#####

with_north_bc(local_bcs, ::Nothing) = local_bcs
with_north_bc(local_bcs, north_bc) = FieldBoundaryConditions(; west=local_bcs.west,
                                                               east=local_bcs.east,
                                                               south=local_bcs.south,
                                                               north=north_bc,
                                                               top=local_bcs.top,
                                                               bottom=local_bcs.bottom)

inject_halo_communication_boundary_conditions(::Nothing, loc, grid::MPITripolarGridOfSomeKind) = nothing
inject_halo_communication_boundary_conditions(::Missing, loc, grid::MPITripolarGridOfSomeKind) = missing

function inject_halo_communication_boundary_conditions(global_bcs, loc, grid::MPITripolarGridOfSomeKind)
    arch = architecture(grid)
    local_bcs = inject_halo_communication_boundary_conditions(global_bcs, loc, arch.local_rank, arch.connectivity, topology(grid))
    north_bc = north_zipper_bc(topology(grid, 2)(), global_bcs.north, loc, grid)
    return with_north_bc(local_bcs, north_bc)
end

# Extension of the constructor for a `Field` on a distributed tripolar grid.
# The north boundary is a zipper with a sign that depends on the location of the field
# (revert the value of the halos if on edges, keep it if on nodes or centers).
function Field(loc::Tuple{<:LX, <:LY, <:LZ}, grid::MPITripolarGridOfSomeKind, data, global_bcs, indices::Tuple, op, status) where {LX, LY, LZ}
    indices = validate_indices(indices, loc, grid)
    validate_field_data(loc, data, grid, indices)
    validate_boundary_conditions(loc, grid, global_bcs)
    local_bcs = inject_halo_communication_boundary_conditions(global_bcs, loc, grid)
    buffers = communication_buffers(grid, data, local_bcs, (LX(), LY(), LZ()))
    return Field{LX, LY, LZ}(grid, data, local_bcs, indices, op, status, buffers)
end

# Reconstruction the global tripolar grid for visualization purposes
function DistributedComputations.reconstruct_global_grid(grid::MPITripolarGrid)

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
                        z,
                        fold_topology = fold_topology(grid.conformal_mapping))
end

function Grids.with_halo(new_halo, old_grid::MPITripolarGrid)

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
                        z,
                        fold_topology = fold_topology(old_grid.conformal_mapping))
end

#####
##### Extend worksize for distributed FPivot grids (matches RFTRG worksize for serial FPivot)
#####

const DistributedFPivotTopology = Union{LeftConnectedRightFaceFolded, LeftConnectedRightFaceConnected}
const DRFTRG = Union{MPITripolarGrid{<:Any, <:Any, <:DistributedFPivotTopology},
                     ImmersedBoundaryGrid{<:Any, <:Any, <:DistributedFPivotTopology, <:Any, <:MPITripolarGrid}}

Utils.worksize(grid::DRFTRG) = grid.Nx, grid.Ny+1, grid.Nz
