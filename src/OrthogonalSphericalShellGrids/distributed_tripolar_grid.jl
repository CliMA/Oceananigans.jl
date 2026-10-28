using Oceananigans.BoundaryConditions: DefaultBoundaryCondition, pivot_shift
using Oceananigans.DistributedComputations:
    DistributedComputations,
    Distributed,
    HaloImage,
    RankConnectivity,
    with_connectivity,
    local_ranges,
    child_architecture,
    ranks,
    concatenate_local_sizes

using Oceananigans.Grids: topology, FullyConnected, RightCenterFolded, RightFaceFolded,
    LeftConnectedRightFaceFolded, LeftConnectedRightFaceConnected, DistributedFoldedTopology
using Oceananigans.DistributedComputations: insert_connected_topology
using Oceananigans.Utils: Utils

const DistributedTripolarGrid{FT, TX, TY, TZ, CZ, CC, FC, CF, FF, Arch} =
    OrthogonalSphericalShellGrid{FT, TX, TY, TZ, CZ, <:Tripolar, CC, FC, CF, FF, <:Distributed{<:Union{CPU, GPU}}}

const MPITripolarGrid{FT, TX, TY, TZ, CZ, CC, FC, CF, FF, Arch} = OrthogonalSphericalShellGrid{FT, TX, TY, TZ, CZ, <:Tripolar, CC, FC, CF, FF, <:Distributed{<:Union{CPU, GPU}}}
const MPITripolarGridOfSomeKind{TY} = Union{MPITripolarGrid{<:Any, <:Any, TY},
    ImmersedBoundaryGrid{<:Any, <:Any, TY, <:Any, <:MPITripolarGrid}}

const DistTRG = MPITripolarGridOfSomeKind{<:DistributedFoldedTopology}

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

    arch = with_connectivity(RankConnectivity(arch.partition, (Nx, Ny, Nz), topology(global_grid), (Hx, Hy, Hz)), arch)

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
##### Halo images across the fold
#####

# Below the fold the halo cells are copies, beyond the fold they copy their mirror image (i, j) ↦ (ox - i, oy - j)
function DistributedComputations.halo_images(grid::MPITripolarGridOfSomeKind, loc)
    connectivity = architecture(grid).connectivity
    Hx, Hy, _ = halo_size(grid)
    Nx, Ny, _ = connectivity.size
    TY = connectivity.topology[2]
    Rx = architecture(grid).ranks[1]

    columns = 1 - Hx : Nx + Hx + 1
    rows = 1 - Hy : Ny + Hy + 1

    # A fold held by a single rank is filled by the zipper
    Rx == 1 && return [HaloImage((columns, rows), 1, 0, 0)]

    ox = Nx + 1 + pivot_shift(fold_pivot(grid)) + (loc[1] isa Face)
    oy = 2Ny + (TY === RightFaceFolded) + (loc[2] isa Face)

    # The fold maps row oy ÷ 2 onto itself when oy is even
    rows_below = first(rows) : cld(oy, 2) - 1
    rows_above = oy ÷ 2 + 1 : last(rows)
    fold_row = cld(oy, 2) : oy ÷ 2

    # The redundant part of the fold row copies its mirror image: i > i′, or i > Nx ÷ 2 with an F pivot
    redundant = filter(i -> TY === RightFaceFolded ? i > Nx ÷ 2 : i > mod1(ox - i, Nx), 1:Nx)
    i₁, i₂ = first(redundant), last(redundant)

    images = [HaloImage((columns, rows_below), 1, 0, 0), HaloImage((columns, rows_above), -1, ox, oy)]

    for q in (-Nx, 0, Nx)
        push!(images, HaloImage((1 + q : i₁ - 1 + q, fold_row), 1, 0, 0))
        push!(images, HaloImage((i₁ + q : i₂ + q, fold_row), -1, ox, oy))
        push!(images, HaloImage((i₂ + 1 + q : Nx + q, fold_row), 1, 0, 0))
    end

    return images
end

#####
##### North fold boundary conditions
#####

# The default north boundary of a rank on the fold is the zipper with the sign of the field. Where the fold spans
# several ranks, the halo communication replaces it and carries its sign.
BoundaryConditions.regularize_boundary_condition(::DefaultBoundaryCondition, grid::DistTRG, loc, dim, bound,
                                                 prognostic_names, sign) = north_fold_boundary_condition(grid, sign)

# User-supplied BC on a distributed fold-north rank → pass through (Field validates later).
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
