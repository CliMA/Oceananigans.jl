using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids: AbstractGrid

using KernelAbstractions: @kernel, @index

import Oceananigans.Grids: get_active_column_map, get_active_cells_map

# REMEMBER: since the active map is stripped out of the grid when `Adapt`ing to the GPU,
# The following types cannot be used to dispatch in kernels!!!

# An IBG with a single interior active cells map that includes the whole :xyz domain
const WholeActiveCellsMapIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray}

# An IBG with an interior active cells map subdivided in 5 different sub-maps.
# Only used (for the moment) in the case of distributed architectures where the boundary adjacent region
# has to be computed separately, these maps hold the active region in the "halo-independent" part of the domain
# (; halo_independent_cells), and the "halo-dependent" regions in the west, east, north, and south, respectively
const SplitActiveCellsMapIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:NamedTuple}

@inline get_active_column_map(grid::ActiveZColumnsIBG) = grid.active_z_columns

@inline get_active_cells_map(grid::WholeActiveCellsMapIBG, ::Val{:interior}) = grid.interior_active_cells
@inline get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:interior}) = grid.interior_active_cells.halo_independent_cells
@inline get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:west})     = grid.interior_active_cells.west_halo_dependent_cells
@inline get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:east})     = grid.interior_active_cells.east_halo_dependent_cells
@inline get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:south})    = grid.interior_active_cells.south_halo_dependent_cells
@inline get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:north})    = grid.interior_active_cells.north_halo_dependent_cells
@inline get_active_cells_map(grid::ActiveZColumnsIBG,      ::Val{:surface})  = grid.active_z_columns

"""
    linear_index_to_tuple(idx, map, grid)

Converts a linear index to a tuple of indices based on the given map and grid.

# Arguments
- `idx`: The linear index to convert.
- `active_cells_map`: The map containing the N-dimensional index of the active cells

# Returns
A tuple of indices corresponding to the linear index.
"""
@inline linear_index_to_tuple(idx, active_cells_map) = @inbounds Base.map(Int, active_cells_map[idx])

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

    for k in 1:size(grid, 3)
        interior_indices = findall(on_architecture(CPU(), interior(active_cells_field, :, :, k:k)))
        interior_indices = convert_interior_indices(interior_indices, k, IndicesType)
        active_indices   = vcat(active_indices, interior_indices)
        GC.gc()
    end

    return active_indices
end

function convert_interior_indices(interior_indices, k, IndicesType)
    interior_indices =   getproperty.(interior_indices, :I)
    interior_indices = add_3rd_index.(interior_indices, k) |> Array{IndicesType}
    return interior_indices
end

@inline add_3rd_index(ij::Tuple, k) = (ij[1], ij[2], k)

# In case of a serial grid, the interior computations are performed over the whole three-dimensional
# domain. Therefore, the `interior_active_cells` field contains the indices of all the active cells in
# the range 1:Nx, 1:Ny and 1:Nz (i.e., we construct the map with parameters :xyz)
build_active_cells_map(grid, ib) = serially_build_active_cells_map(grid, ib; parameters = :xyz)

# If we eventually want to perform also barotropic step, `w` computation and `p`
# computation only on active `columns`
function build_active_z_columns(grid, ib)
    field = compute_active_z_columns(grid, ib)
    field_interior = on_architecture(CPU(), interior(field, :, :, 1))

    full_indices = findall(field_interior)

    Nx, Ny, _ = size(grid)
    # Reduce the size of the active_cells_map (originally a tuple of Int64)
    N = max(Nx, Ny)
    IntType = N > MAXUInt8 ? (N > MAXUInt16 ? (N > MAXUInt32 ? UInt64 : UInt32) : UInt16) : UInt8
    columns_map = getproperty.(full_indices, Ref(:I)) .|> Tuple{IntType, IntType}
    columns_map = on_architecture(architecture(grid), columns_map)

    return columns_map
end

