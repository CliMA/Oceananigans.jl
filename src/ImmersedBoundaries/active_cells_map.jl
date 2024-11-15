using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids: AbstractGrid

using KernelAbstractions: @kernel, @index

import Oceananigans.Grids: retrieve_surface_active_cells_map, retrieve_interior_active_cells_map

# REMEMBER: since the active map is stripped out of the grid when `Adapt`ing to the GPU, 
# The following types cannot be used to dispatch in kernels!!!

# An IBG with a single interior active cells map that includes the whole :xyz domain
const WholeActiveCellsMapIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray}

# An IBG with an interior active cells map subdivided in 5 different sub-maps.
# Only used (for the moment) in the case of distributed architectures where the boundary adjacent region 
# has to be computed separately, these maps hold the active region in the "halo-independent" part of the domain
# (; halo_independent_cells), and the "halo-dependent" regions in the west, east, north, and south, respectively
const SplitActiveCellsMapIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:NamedTuple}

"""
A constant representing an immersed boundary grid, where interior active cells are mapped to linear indices in grid.interior_active_cells
"""
const ActiveCellsIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Union{AbstractArray, NamedTuple}}

"""
A constant representing an immersed boundary grid, where active columns in the Z-direction are mapped to linear indices in grid.active_z_columns
"""
const ActiveZColumnsIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray}

@inline retrieve_surface_active_cells_map(grid::ActiveZColumnsIBG) = grid.active_z_columns

@inline retrieve_interior_active_cells_map(grid::WholeActiveCellsMapIBG, ::Val{:interior}) = grid.interior_active_cells
@inline retrieve_interior_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:interior}) = grid.interior_active_cells.halo_independent_cells
@inline retrieve_interior_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:west})     = grid.interior_active_cells.west_halo_dependent_cells
@inline retrieve_interior_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:east})     = grid.interior_active_cells.east_halo_dependent_cells
@inline retrieve_interior_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:south})    = grid.interior_active_cells.south_halo_dependent_cells
@inline retrieve_interior_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:north})    = grid.interior_active_cells.north_halo_dependent_cells
@inline retrieve_interior_active_cells_map(grid::ActiveZColumnsIBG,      ::Val{:surface})  = grid.active_z_columns

"""
    active_linear_index_to_tuple(idx, map, grid)

Converts a linear index to a tuple of indices based on the given map and grid.

# Arguments
- `idx`: The linear index to convert.
- `active_cells_map`: The map containing the N-dimensional index of the active cells

# Returns
A tuple of indices corresponding to the linear index.
"""
@inline active_linear_index_to_tuple(idx, active_cells_map) = @inbounds Base.map(Int, active_cells_map[idx])

function ImmersedBoundaryGrid(grid, ib; active_cells_map::Bool = true) 

    ibg = ImmersedBoundaryGrid(grid, ib)
    TX, TY, TZ = topology(ibg)
    
    # Create the cells map on the CPU, then switch it to the GPU
    if active_cells_map 
        interior_map = map_interior_active_cells(ibg)
        column_map   = map_active_z_columns(ibg)
    else
        interior_map = nothing
        column_map  = nothing
    end

    return ImmersedBoundaryGrid{TX, TY, TZ}(ibg.underlying_grid, 
                                            ibg.immersed_boundary, 
                                            interior_map,
                                            column_map)
end

with_halo(halo, ibg::ActiveCellsIBG) =
    ImmersedBoundaryGrid(with_halo(halo, ibg.underlying_grid), ibg.immersed_boundary; active_cells_map = true)

@inline active_cell(i, j, k, ibg) = !immersed_cell(i, j, k, ibg)
@inline active_column(i, j, k, grid, column) = column[i, j, k] != 0

@kernel function _set_active_indices!(active_cells_field, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds active_cells_field[i, j, k] = active_cell(i, j, k, grid)
end

function compute_interior_active_cells(ibg; parameters = :xyz)
    active_cells_field = Field{Center, Center, Center}(ibg, Bool)
    fill!(active_cells_field, false)
    launch!(architecture(ibg), ibg, parameters, _set_active_indices!, active_cells_field, ibg)
    return active_cells_field
end

function compute_active_z_columns(ibg)
    one_field = OneField(Int)
    condition = NotImmersed(truefunc)
    mask = 0

    # Compute all the active cells in a z-column using a ConditionalOperation
    conditional_active_cells = ConditionalOperation{Center, Center, Center}(one_field, identity, ibg, condition, mask)
    active_cells_in_column   = sum(conditional_active_cells, dims = 3)

    # Check whether the column ``i, j`` is immersed, which would correspond to `active_cells_in_column[i, j, 1] == 0`
    is_immersed_column = KernelFunctionOperation{Center, Center, Nothing}(active_column, ibg, active_cells_in_column)
    active_z_columns = Field{Center, Center, Nothing}(ibg, Bool)
    set!(active_z_columns, is_immersed_column)

    return active_z_columns
end

# Maximum integer represented by the 
# `UInt8`, `UInt16` and `UInt32` types
const MAXUInt8  = 2^8  - 1
const MAXUInt16 = 2^16 - 1
const MAXUInt32 = 2^32 - 1

"""
    interior_active_indices(ibg; parameters = :xyz)

Compute the indices of the active interior cells in the given immersed boundary grid within the indices
specified by the `parameters` keyword argument

# Arguments
- `ibg`: The immersed boundary grid.
- `parameters`: (optional) The parameters to be used for computing the active cells. Default is `:xyz`.

# Returns
An array of tuples representing the indices of the active interior cells.
"""
function interior_active_indices(ibg; parameters = :xyz)
    active_cells_field = compute_interior_active_cells(ibg; parameters)
    
    N = maximum(size(ibg))
    IntType = N > MAXUInt8 ? (N > MAXUInt16 ? (N > MAXUInt32 ? UInt64 : UInt32) : UInt16) : UInt8
   
    IndicesType = Tuple{IntType, IntType, IntType}

    # Cannot findall on the entire field because we could incur on OOM errors
    # For this reason, we split the computation in vertical levels and `findall` the active indices in 
    # subsequent xy planes, then stitch them back together
    active_indices = IndicesType[]
    active_indices = findall_active_indices!(active_indices, active_cells_field, ibg, IndicesType)
    active_indices = on_architecture(architecture(ibg), active_indices)

    return active_indices
end

# Cannot `findall` on very large grids, so we split the computation in levels.
# This makes the computation a little heavier but avoids OOM errors (this computation
# is performed only once on setup)
function findall_active_indices!(active_indices, active_cells_field, ibg, IndicesType)
    
    for k in 1:size(ibg, 3)
        interior_indices = findall(on_architecture(CPU(), interior(active_cells_field, :, :, k:k)))
        interior_indices = convert_interior_indices(interior_indices, k, IndicesType)
        active_indices = vcat(active_indices, interior_indices)
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
map_interior_active_cells(ibg) = interior_active_indices(ibg; parameters = :xyz)

# If we eventually want to perform also barotropic step, `w` computation and `p` 
# computation only on active `columns`
function map_active_z_columns(ibg)
    active_cells_field = compute_active_z_columns(ibg)
    interior_cells     = on_architecture(CPU(), interior(active_cells_field, :, :, 1))
  
    full_indices = findall(interior_cells)

    Nx, Ny, _ = size(ibg)
    # Reduce the size of the active_cells_map (originally a tuple of Int64)
    N = max(Nx, Ny)
    IntType = N > MAXUInt8 ? (N > MAXUInt16 ? (N > MAXUInt32 ? UInt64 : UInt32) : UInt16) : UInt8
    surface_map = getproperty.(full_indices, Ref(:I)) .|> Tuple{IntType, IntType}
    surface_map = on_architecture(architecture(ibg), surface_map)

    return surface_map
end
