using Oceananigans.Architectures: CPU
using Oceananigans.Fields: Field
using Oceananigans.Grids: Grids, AbstractGrid
using Oceananigans.Utils: Utils, worksize, convert_interior_indices
using KernelAbstractions: @kernel, @index

# REMEMBER: since the active map is stripped out of the grid when `Adapt`ing to the GPU,
# The following types cannot be used to dispatch in kernels!!!

# An IBG with a single interior active cells map that includes the whole :xyz domain
const WholeActiveCellsMapIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray}

# An IBG with an interior active cells map subdivided in 5 different sub-maps.
# Only used (for the moment) in the case of distributed architectures where the boundary adjacent region
# has to be computed separately, these maps hold the active region in the "halo-independent" part of the domain
# (; halo_independent_cells), and the "halo-dependent" regions in the west, east, north, and south, respectively
const SplitActiveCellsMapIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:NamedTuple}

# `get_active_cells_map` returns the precomputed list of active (non-immersed) cell indices
# associated with a given iteration region. The `Val{...}` symbol selects which region.
#
# When a kernel is launched with one of these workspecs (via `launch!(arch, grid, workspec, ...)`)
# and the grid carries the corresponding map, the kernel is rewritten as a one-dimensional
# loop over the entries of the map, so only active (i, j[, k]) tuples are visited and the
# immersed cells are skipped entirely.
#
#   :xyz   -- the full three-dimensional interior. The map is the flat list of active cells
#             in the range 1:Nx, 1:Ny, 1:Nz. Used by any kernel that would otherwise loop
#             over every interior (i, j, k) point (e.g. tendency computations).
#
#   :xy    -- the "surface" interior: one entry per active horizontal column, i.e. only the
#             (i, j) columns that contain at least one active cell. Used by 2D / column-wise
#             kernels (free-surface / barotropic step, vertical integrals, surface bottom-
#             height computations, masking of horizontal slabs, ...) so that fully immersed
#             columns are skipped.
#
# The remaining symbols are only meaningful for `SplitActiveCellsMapIBG`, where the 3D
# interior map is partitioned into a "halo-independent" core and four "halo-dependent"
# boundary strips. This split lets distributed runs overlap MPI halo communication with
# computation: the core can be advanced while halo exchanges are still in flight, and the
# boundary strips are launched only after the matching halo exchange completes.
#
#   :core  -- the halo-independent part of the 3D interior, i.e. cells far enough from the
#             local subdomain edges that their stencils never reach into halo points. For a
#             `WholeActiveCellsMapIBG` (no split, e.g. serial runs) this is just the full
#             interior map and is equivalent to `:xyz`.
#
#   :west  -- halo-dependent strip on the western (-x) edge of the local subdomain.
#   :east  -- halo-dependent strip on the eastern (+x) edge of the local subdomain.
#   :south -- halo-dependent strip on the southern (-y) edge of the local subdomain.
#   :north -- halo-dependent strip on the northern (+y) edge of the local subdomain.
#
# Each of these four strips contains the active cells whose stencils touch the corresponding
# halo region, and must therefore wait for the matching halo exchange before being computed.
@inline Utils.get_active_cells_map(grid::ActiveInteriorIBG,      ::Val{:xyz})   = grid.interior_active_cells
@inline Utils.get_active_cells_map(grid::ActiveZColumnsIBG,      ::Val{:xy})    = grid.active_z_columns
@inline Utils.get_active_cells_map(grid::WholeActiveCellsMapIBG, ::Val{:core})  = grid.interior_active_cells
@inline Utils.get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:core})  = grid.interior_active_cells.halo_independent_cells
@inline Utils.get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:west})  = grid.interior_active_cells.west_halo_dependent_cells
@inline Utils.get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:east})  = grid.interior_active_cells.east_halo_dependent_cells
@inline Utils.get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:south}) = grid.interior_active_cells.south_halo_dependent_cells
@inline Utils.get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:north}) = grid.interior_active_cells.north_halo_dependent_cells

"""
    linear_index_to_tuple(idx, map, grid)

Converts a linear index to a tuple of indices based on the given map and grid.

# Arguments
- `idx`: The linear index to convert.
- `active_cells_map`: The map containing the N-dimensional index of the active cells

# Returns
A tuple of indices corresponding to the linear index.
"""
@inline linear_index_to_tuple(idx, active_cells_map) = @inbounds active_cells_map[idx]

#=
@inline function linear_index_to_tuple(idx, active_cells_map::Tuple{<:Any, <:Any, <:Any})
    @inbounds begin
        i = active_cells_map[1][idx]
        j = active_cells_map[2][idx]
        k = active_cells_map[3][idx]
    end
    return i, j, k
end

@inline function linear_index_to_tuple(idx, active_cells_map::Tuple{<:Any, <:Any})
    @inbounds begin
        i = active_cells_map[1][idx]
        j = active_cells_map[2][idx]
    end
    return i, j
end
=#

@inline active_cell(i, j, k, grid, ib) = !immersed_cell(i, j, k, grid, ib)

@kernel function _compute_active_cells!(active_cells_field, grid, ib)
    i, j, k = @index(Global, NTuple)
    @inbounds active_cells_field[i, j, k] = active_cell(i, j, k, grid, ib)
end

function compute_active_cells(grid, ib; parameters = :xyz)
    active_cells_field = Field{Center, Center, Center}(grid, Bool)
    fill!(active_cells_field, false)
    launch!(architecture(grid), grid, parameters, _compute_active_cells!, active_cells_field, grid, ib)
    return active_cells_field
end

@kernel function _compute_active_z_columns!(active_z_columns, grid, ib)
    i, j = @index(Global, NTuple)
    active_column = false
    for k in 1:size(grid, 3)
        active_column = active_column | active_cell(i, j, k, grid, ib)
    end
    @inbounds active_z_columns[i, j, 1] = active_column
end

function compute_active_z_columns(grid, ib)
    active_z_columns = Field{Center, Center, Nothing}(grid, Bool)
    fill!(active_z_columns, false)

    # Compute the active cells in the column
    launch!(architecture(grid), grid, :xy, _compute_active_z_columns!, active_z_columns, grid, ib)

    return active_z_columns
end

# Maximum integer represented by the
# `UInt8`, `UInt16` and `UInt32` types
const MAXUInt8  = 2^8  - 1
const MAXUInt16 = 2^16 - 1
const MAXUInt32 = 2^32 - 1

"""
    serially_build_active_cells_map(grid, ib; parameters = :xyz)

Compute the indices of the active interior cells in the given immersed boundary grid within the indices
specified by the `parameters` keyword argument

# Arguments
- `grid`: The underlying grid.
- `ib`: The immersed boundary.
- `parameters`: (optional) The parameters to be used for computing the active cells. Default is `:xyz`.

# Returns
An array of tuples representing the indices of the active interior cells.
"""
function serially_build_active_cells_map(grid, ib; parameters = :xyz)
    active_cells_field = compute_active_cells(grid, ib; parameters)

    N = maximum(size(grid))
    IntType = N > MAXUInt8 ? (N > MAXUInt16 ? (N > MAXUInt32 ? UInt64 : UInt32) : UInt16) : UInt8

    IndicesType = Tuple{IntType, IntType, IntType}

    # Cannot findall on the entire field because we could incur on OOM errors
    # For this reason, we split the computation in vertical levels and `findall` the active indices in
    # subsequent xy planes, then stitch them back together
    active_indices = IndicesType[]
    active_indices = findall_active_indices!(active_indices, active_cells_field, grid, IndicesType)
    active_indices = on_architecture(architecture(grid), active_indices)

    return active_indices
end

# Cannot `findall` on very large grids, so we split the computation in levels.
# This makes the computation a little heavier but avoids OOM errors (this computation
# is performed only once on setup)
function findall_active_indices!(active_indices, active_cells_field, grid, IndicesType)
    Wx, Wy, Wz = worksize(grid)
    for k in 1:Wz
        interior_indices = findall(on_architecture(CPU(), view(active_cells_field.data, 1:Wx, 1:Wy, k:k)))
        interior_indices = convert_interior_indices(interior_indices, k, IndicesType)
        active_indices   = vcat(active_indices, interior_indices)
        GC.gc()
    end
    return active_indices
end

# In case of a serial grid, the interior computations are performed over the whole three-dimensional
# domain. Therefore, the `interior_active_cells` field contains the indices of all the active cells in
# the range 1:Nx, 1:Ny and 1:Nz (i.e., we construct the map with parameters :xyz)
build_active_cells_map(grid, ib) = serially_build_active_cells_map(grid, ib; parameters = :xyz)

# If we eventually want to perform also barotropic step, `w` computation and `p`
# computation only on active `columns`
function build_active_z_columns(grid, ib)
    field = compute_active_z_columns(grid, ib)
    Wx, Wy, Wz = worksize(grid)
    field_data = on_architecture(CPU(), view(field.data, 1:Wx, 1:Wy, 1))
    full_indices = findall(field_data)

    # Reduce the size of the active_cells_map (originally a tuple of Int64)
    N = max(Wx, Wy)
    IntType = N > MAXUInt8 ? (N > MAXUInt16 ? (N > MAXUInt32 ? UInt64 : UInt32) : UInt16) : UInt8
    columns_map = getproperty.(full_indices, Ref(:I)) .|> Tuple{IntType, IntType}
    columns_map = on_architecture(architecture(grid), columns_map)

    return columns_map
end
