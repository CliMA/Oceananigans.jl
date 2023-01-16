using Oceananigans
using Oceananigans.Grids: AbstractGrid
import Oceananigans.Utils: only_active_cells_in_worksize, calc_tendency_index
using KernelAbstractions: @kernel, @index

only_active_cells_in_worksize(size, grid::IBG) = min(length(grid.active_cells_map), 256), length(grid.active_cells_map)
@inline calc_tendency_index(idx, grid::IBG)    = Int.(grid.active_cells_map[idx])

const FullIBG        = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing}
const ActiveCellsIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray}

@inline use_only_active_cells(grid::AbstractGrid)   = false
@inline use_only_active_cells(grid::ActiveCellsIBG) = true

function ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib; calculate_active_cells_map = false) where {TX, TY, TZ} 

    # Create the cells map on the CPU, then switch it to the GPU
    if calculate_active_cells_map 
        active_cells_map    = create_cells_map(grid, ib)
        active_cells_map    = arch_array(architecture(grid), active_cells_map)
    else
        active_cells_map = nothing
    end

    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib, active_cells_map)
end

active_cell(i, j, k, grid, ib) = !immersed_cell(i, j, k, grid, ib)

function compute_active_cells(grid, ib)
    is_immersed_operation = KernelFunctionOperation{Center, Center, Center}(active_cell, grid; computed_dependencies = (ib, ))
    active_cells_field = Field{Center, Center, Center}(grid, Bool)
    set!(active_cells_field, is_immersed_operation)
    return active_cells_field
end

const MAXUInt8  = 2^8  - 1
const MAXUInt16 = 2^16 - 1
const MAXUInt32 = 2^32 - 1

function create_cells_map(grid, ib)
    active_cells_field = compute_active_cells(grid, ib)
    full_indices       = arch_array(CPU(), findall(interior(active_cells_field)))
    
    # Reduce the size of the active_cells_map (originally a tuple of Int64)
    N = maximum(size(grid))
    Type = N > MAXUInt8 ? (N > MAXUInt16 ? (N > MAXUInt32 ? UInt64 : UInt32) : UInt16) : UInt8
    smaller_indices = getproperty.(full_indices, Ref(:I)) .|> Tuple{Type, Type, Type}
    
    return smaller_indices
end