using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids: AbstractGrid

using KernelAbstractions: @kernel, @index

import Oceananigans.Grids: retrieve_surface_active_cells_map, retrieve_interior_active_cells_map
import Oceananigans.Utils: active_cells_work_layout

using Oceananigans.Solvers: solve_batched_tridiagonal_system_z!, ZDirection
using Oceananigans.DistributedComputations: DistributedGrid

import Oceananigans.Solvers: solve_batched_tridiagonal_system_kernel!

const DistributedActiveCellsIBG   = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:DistributedGrid, <:Any, <:NamedTuple} # Cannot be used to dispatch in kernels!!!

# An IBG with an active cells map that includes the whole :xyz domain
const ArrayActiveCellsMapIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray}

# An IBG with an active cells map subdivided in (; interior, west, east, north, south)
# Only used (for the moment) in the case of distributed architectures where the boundary adjacent region 
# has to be computed separately, these maps hold the active region in the "halo-independent" part of the domain
# (; interior), and the "halo-dependent" regions in the west, east, north, and south, respectively
const NamedTupleActiveCellsMapIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:NamedTuple}

"""
A constant representing an immersed boundary grid, where interior active cells are mapped to linear indices in grid.interior_active_cells
"""
const ActiveCellsIBG = Union{DistributedActiveCellsIBG, ArrayActiveCellsIBG, NamedTupleActiveCellsIBG}

"""
A constant representing an immersed boundary grid, where active columns in the Z-direction are mapped to linear indices in grid.active_z_columns
"""
const ActiveZColumnsIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray}

@inline retrieve_surface_active_cells_map(grid::ActiveZColumnsIBG) = grid.active_z_columns

@inline retrieve_interior_active_cells_map(grid::ArrayActiveCellsMapIBG,      ::Val{:interior}) = grid.interior_active_cells
@inline retrieve_interior_active_cells_map(grid::NamedTupleActiveCellsMapIBG, ::Val{:interior}) = grid.interior_active_cells.halo_independent_cells
@inline retrieve_interior_active_cells_map(grid::NamedTupleActiveCellsMapIBG, ::Val{:west})     = grid.interior_active_cells.west_halo_dependent_cells
@inline retrieve_interior_active_cells_map(grid::NamedTupleActiveCellsMapIBG, ::Val{:east})     = grid.interior_active_cells.east_halo_dependent_cells
@inline retrieve_interior_active_cells_map(grid::NamedTupleActiveCellsMapIBG, ::Val{:south})    = grid.interior_active_cells.south_halo_dependent_cells
@inline retrieve_interior_active_cells_map(grid::NamedTupleActiveCellsMapIBG, ::Val{:north})    = grid.interior_active_cells.north_halo_dependent_cells
@inline retrieve_interior_active_cells_map(grid::ActiveZColumnsIBG,           ::Val{:surface})  = grid.active_z_columns

"""
    active_cells_work_layout(group, size, map_type, grid)

Compute the work layout for active cells based on the given map type and grid.

# Arguments
- `group`: The previous workgroup.
- `size`: The previous worksize.
- `active_cells_map`: The map containing the index of the active cells

# Returns
- A tuple `(workgroup, worksize)` representing the work layout for active cells.
"""
@inline active_cells_work_layout(group, size, active_cells_map) = min(length(active_cells_map), 256), length(active_cells_map)

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
    one_field = ConditionalOperation{Center, Center, Center}(OneField(Int), identity, ibg, NotImmersed(truefunc), 0)
    column    = sum(one_field, dims = 3)
    is_immersed_column = KernelFunctionOperation{Center, Center, Nothing}(active_column, ibg, column)
    active_cells_field = Field{Center, Center, Nothing}(ibg, Bool)
    set!(active_cells_field, is_immersed_column)
    return active_cells_field
end

const MAXUInt8  = 2^8  - 1
const MAXUInt16 = 2^16 - 1
const MAXUInt32 = 2^32 - 1

"""
    interior_active_indices(ibg; parameters = :xyz)

Compute the indices of the active interior cells in the given immersed boundary grid.

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

    # Cannot findall on the entire field because we incur on OOM errors
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

@inline add_3rd_index(t::Tuple, k) = (t[1], t[2], k) 

# In case of a serial grid, the interior computations are performed over the whole three-dimensional
# domain. Therefore, the `interior_active_cells` field contains the indices of all the active cells in 
# the active domain from 1:Nz, 1:Ny and 1:Nz (i.e.,  we construct the map with parameters :xyz)
map_interior_active_cells(ibg) = interior_active_indices(ibg; parameters = :xyz)

# In case of a `DistributedGrid` we want to have different maps depending on the partitioning of the domain:
#
# If we partition the domain in the x-direction, we typically want to have the option to split three-dimensional 
# kernels in a `halo-independent` part that includes Hx+1:Nx-Hx, 1:Ny, 1:Nz and two `halo-dependent` computations: a west
# one spanning 1:Hx, 1:Ny, 1:Nz and a east one spanning Nx-Hx+1:Nx, 1:Ny, 1:Nz. For this reason we need three different maps,
# one containing the `halo_independent` part, a `west` map and an `east` map. 
# For the same reason we need to construct `south` and `north` maps if we partition the domain in the y-direction.
# Therefore, the `interior_active_cells` in this case is a `NamedTuple` containing 5 elements.
function map_interior_active_cells(ibg::ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:DistributedGrid})

    arch = architecture(ibg)
    Rx, Ry, _  = arch.ranks
    Tx, Ty, _  = topology(ibg)
    Nx, Ny, Nz = size(ibg)
    Hx, Hy, _  = halo_size(ibg)
    
    x_boundary = (Hx, Ny, Nz)
    y_boundary = (Nx, Hy, Nz)
         
    left_offsets    = (0,  0,  0)
    right_x_offsets = (Nx-Hx, 0,     0)
    right_y_offsets = (0,     Ny-Hy, 0)

    include_west  = !isa(ibg, XFlatGrid) && (Rx != 1) && !(Tx == RightConnected)
    include_east  = !isa(ibg, XFlatGrid) && (Rx != 1) && !(Tx == LeftConnected)
    include_south = !isa(ibg, YFlatGrid) && (Ry != 1) && !(Ty == RightConnected)
    include_north = !isa(ibg, YFlatGrid) && (Ry != 1) && !(Ty == LeftConnected)

    west_halo_dependent_cells  = include_west  ? interior_active_indices(ibg; parameters = KernelParameters(x_boundary, left_offsets))    : nothing
    east_halo_dependent_cells  = include_east  ? interior_active_indices(ibg; parameters = KernelParameters(x_boundary, right_x_offsets)) : nothing
    south_halo_dependent_cells = include_south ? interior_active_indices(ibg; parameters = KernelParameters(y_boundary, left_offsets))    : nothing
    north_halo_dependent_cells = include_north ? interior_active_indices(ibg; parameters = KernelParameters(y_boundary, right_y_offsets)) : nothing
    
    nx = Rx == 1 ? Nx : (Tx == RightConnected || Tx == LeftConnected ? Nx - Hx : Nx - 2Hx)
    ny = Ry == 1 ? Ny : (Ty == RightConnected || Ty == LeftConnected ? Ny - Hy : Ny - 2Hy)

    ox = Rx == 1 || Tx == RightConnected ? 0 : Hx
    oy = Ry == 1 || Ty == RightConnected ? 0 : Hy
     
    halo_independent_cells = interior_active_indices(ibg; parameters = KernelParameters((nx, ny, Nz), (ox, oy, 0)))

    return (; halo_independent_cells, 
              west_halo_dependent_cells, 
              east_halo_dependent_cells, 
              south_halo_dependent_cells, 
              north_halo_dependent_cells)
end

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
