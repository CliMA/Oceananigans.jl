using Oceananigans.BoundaryConditions: DistributedCommunicationBoundaryCondition, ZBC, DCBC, DefaultBoundaryCondition, pivot_shift
using Oceananigans.Fields: validate_indices, validate_field_data
using Oceananigans.DistributedComputations:
    DistributedComputations,
    Distributed,
    RankLayout,
    with_rank_layout,
    local_ranges,
    child_architecture,
    ranks,
    inject_halo_communication_boundary_conditions,
    concatenate_local_sizes,
    communication_buffers

using Oceananigans.Grids: topology, FullyConnected,
    RightCenterFolded, RightFaceFolded,
    LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded,
    LeftConnectedRightCenterConnected, LeftConnectedRightFaceConnected,
    SlabFoldedTopology, PencilFoldedTopology, DistributedFoldedTopology
using Oceananigans.DistributedComputations: insert_connected_topology
using Oceananigans.Utils: Utils

import Oceananigans.Fields: Field, validate_indices, validate_boundary_conditions
import Oceananigans.DistributedComputations: inject_halo_communication_boundary_conditions

const DistributedTripolarGrid{FT, TX, TY, TZ, CZ, CC, FC, CF, FF, Arch} =
    OrthogonalSphericalShellGrid{FT, TX, TY, TZ, CZ, <:Tripolar, CC, FC, CF, FF, <:Distributed{<:Union{CPU, GPU}}}

const MPITripolarGrid{FT, TX, TY, TZ, CZ, CC, FC, CF, FF, Arch} = OrthogonalSphericalShellGrid{FT, TX, TY, TZ, CZ, <:Tripolar, CC, FC, CF, FF, <:Distributed{<:Union{CPU, GPU}}}
const MPITripolarGridOfSomeKind{TY} = Union{MPITripolarGrid{<:Any, <:Any, TY},
    ImmersedBoundaryGrid{<:Any, <:Any, TY, <:Any, <:MPITripolarGrid}}

const SlabTRG   = MPITripolarGridOfSomeKind{<:SlabFoldedTopology}
const PencilTRG = MPITripolarGridOfSomeKind{<:PencilFoldedTopology}
const DistTRG   = MPITripolarGridOfSomeKind{<:DistributedFoldedTopology}

"""
    TripolarGrid(arch::Distributed, FT::DataType = Float64; halo = (4, 4, 4), kwargs...)

Construct a tripolar grid on a distributed `arch`itecture.

!!! compat "Supported partitionings"

    The grid can be partitioned in `y` only, e.g., `Distributed(CPU(), partition=Partition(1, 4))`,
    or in both `x` and `y`, e.g., `Distributed(CPU(), partition=Partition(3, 4))`.
    Partitioning only in `x`, e.g., `Distributed(CPU(), partition=Partition(4))`, is _not_ supported.
"""
function TripolarGrid(arch::Distributed, FT::DataType=Float64;
                      halo=(4, 4, 4),
                      kwargs...)

    # We build the global grid on a CPU architecture, in order to split it easily
    global_grid = TripolarGrid(CPU(), FT; halo, kwargs...)
    return distribute_tripolar_grid(arch, global_grid)
end

"""
    distribute_tripolar_grid(arch, global_grid)

Return the slice of the global tripolar `global_grid` owned by `arch`'s rank. The fold pairs points
across the northern seam, so a tripolar grid is built globally and then partitioned. See
[`TripolarGrid`](@ref) for the supported partitionings. On a serial `arch` the whole grid is returned,
moved to that architecture.
"""
distribute_tripolar_grid(arch, global_grid) = on_architecture(arch, global_grid)

function distribute_tripolar_grid(arch::Distributed, global_grid)
    Rx, Ry, _ = ranks(arch.partition)

    if Rx != 1 && Ry == 1
        throw(ArgumentError("an x-only partitioning is not supported by TripolarGrid, use a y or x-y pencil partitioning"))
    end

    Nx, Ny, Nz = size(global_grid)
    Hx, Hy, Hz = halo_size(global_grid)

    layout = RankLayout(arch.partition, (Nx, Ny, Nz), topology(global_grid), pivot_shift(fold_pivot(global_grid)))
    arch = with_rank_layout(arch, layout)

    i, j, _ = local_ranges(arch)
    irange = first(i)-Hx:last(i)+Hx
    jrange = first(j)-Hy:last(j)+Hy

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

    LX = length(i) == Nx ? Periodic : FullyConnected
    rx, ry, _ = arch.local_index
    LY = insert_connected_topology(topology(global_grid, 2), Ry, ry, Rx, rx)
    nx = length(i)
    ny = length(j)

    z = on_architecture(arch, global_grid.z)
    radius = global_grid.radius

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

#####
##### North fold BC Regularization
#####

# We add `regularize_boundary_condition` methods that pass the zipper BC sign as an extra arg.

# DefaultBC on a slab fold-north rank → local `Zipper` with sign
BoundaryConditions.regularize_boundary_condition(::DefaultBoundaryCondition, grid::SlabTRG, loc, dim, bound, prognostic_names, sign) =
    north_fold_boundary_condition(grid, sign)

# DefaultBC on a pencil fold-north rank → distributed communication BC with the sign of the fold
BoundaryConditions.regularize_boundary_condition(::DefaultBoundaryCondition, grid::PencilTRG, loc, dim, bound,
                                                 prognostic_names, sign) = DistributedCommunicationBoundaryCondition(sign)

# User-supplied BC on either distributed fold-north rank → pass through (Field validates later).
# `bc::BoundaryCondition` mirrors the existing BoundaryConditions.jl:244 method signature, so
# this is strictly more specific and disambiguates.
BoundaryConditions.regularize_boundary_condition(bc::BoundaryCondition, grid::DistTRG, loc, dim, bound, prognostic_names, sign) = bc

# Only to solve the ambiguities (this method should never be used)
BoundaryConditions.regularize_boundary_condition(bc::BoundaryConditions.RBC, grid::DistTRG, loc, dim, bound, prognostic_names, sign) = bc
BoundaryConditions.regularize_boundary_condition(bc::BoundaryConditions.TRVBC, grid::DistTRG, loc, dim, bound, prognostic_names, sign) = bc

# Non-fold distributed ranks: no specific 7-arg method — the generic `args...`-accepting
# methods in BoundaryConditions (lines 244-254) take over and drop the extra `sign` arg.

function BoundaryConditions.regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                                                 grid::MPITripolarGridOfSomeKind,
                                                                 field_name::Symbol,
                                                                 prognostic_names=nothing)

    loc  = assumed_field_location(field_name)
    sign = field_name == :u || field_name == :v ||
           field_name == :U || field_name == :V ? -1 : 1

    west   = regularize_boundary_condition(bcs.west,   grid, loc, 1, LeftBoundary,  prognostic_names)
    east   = regularize_boundary_condition(bcs.east,   grid, loc, 1, RightBoundary, prognostic_names)
    south  = regularize_boundary_condition(bcs.south,  grid, loc, 2, LeftBoundary,  prognostic_names)
    north  = regularize_boundary_condition(bcs.north,  grid, loc, 2, RightBoundary, prognostic_names, sign)
    bottom = regularize_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top    = regularize_boundary_condition(bcs.top,    grid, loc, 3, RightBoundary, prognostic_names)

    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

#####
##### Dispatch on (y-topology, north_bc) to determine the north zipper BC.
#####

zipper_sign(bc::ZBC)  = bc.condition
zipper_sign(bc::DCBC) = bc.condition

# Non-fold topologies for non-fold ranks: no override
north_zipper_bc(topo, north_bc, loc, grid) = nothing

north_zipper_bc(::SlabFoldedTopology, ::Nothing, loc, grid) = nothing
north_zipper_bc(::PencilFoldedTopology, ::Nothing, loc, grid) = nothing

# Distributed slab fold-north rank: local Zipper BC (sign from incoming BC)
north_zipper_bc(::SlabFoldedTopology, north_bc, loc, grid) = north_fold_boundary_condition(grid, zipper_sign(north_bc))

# Distributed pencil fold-north rank (y-topology is `Connected` but carries the fold via MPI):
# a distributed communication BC with the sign of the fold
north_zipper_bc(::PencilFoldedTopology, north_bc, loc, grid) = DistributedCommunicationBoundaryCondition(zipper_sign(north_bc))

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
    local_bcs = inject_halo_communication_boundary_conditions(global_bcs, loc, arch.local_rank, arch.connectivity)
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
                        fold_topology = fold_topology(grid.conformal_mapping),
                        pivot = fold_pivot(grid.conformal_mapping))
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
                        fold_topology = fold_topology(old_grid.conformal_mapping),
                        pivot = fold_pivot(old_grid.conformal_mapping))
end

#####
##### Extend worksize for distributed FPivot grids (matches RFTRG worksize for serial FPivot)
#####

const DistributedFPivotTopology = Union{LeftConnectedRightFaceFolded, LeftConnectedRightFaceConnected}
const DRFTRG = Union{MPITripolarGrid{<:Any, <:Any, <:DistributedFPivotTopology},
                     ImmersedBoundaryGrid{<:Any, <:Any, <:DistributedFPivotTopology, <:Any, <:MPITripolarGrid}}

Utils.worksize(grid::DRFTRG) = grid.Nx, grid.Ny+1, grid.Nz
