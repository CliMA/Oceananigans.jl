using Oceananigans
import Oceananigans.Utils: only_active_cells_in_worksize, calc_tendency_index
using KernelAbstractions: @kernel, @index

only_active_cells_in_worksize(size, grid::IBG) = min(length(grid.wet_cells_map), 256), length(grid.wet_cells_map)

@inline calc_tendency_index(idx, i, j, k, grid::IBG) = Int.(grid.wet_cells_map[idx])

function ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib; calculate_wet_cell_map = false) where {TX, TY, TZ} 

    # Create the cells map on the CPU, then switch it to the GPU
    if calculate_wet_cell_map 
        wet_cells_map    = create_cells_map(grid, ib)
        wet_cells_map    = arch_array(architecture(grid), wet_cells_map)
    else
        wet_cells_map = nothing
    end

    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib, wet_cells_map)
end

wet_cell(i, j, k, grid, ib) = !immersed_cell(i, j, k, grid, ib)

function compute_wet_cells(grid, ib)
    is_immersed_operation = KernelFunctionOperation{Center, Center, Center}(wet_cell, grid; computed_dependencies = (ib, ))
    wet_cells_field = Field{Center, Center, Center}(grid, Bool)
    set!(wet_cells_field, is_immersed_operation)
    return wet_cells_field
end

const MAXUInt8  = 2^8  - 1
const MAXUInt16 = 2^16 - 1
const MAXUInt32 = 2^32 - 1

function create_cells_map(grid, ib)
    wet_cells_field = compute_wet_cells(grid, ib)
    full_indices    = arch_array(CPU(), findall(interior(wet_cells_field)))
    
    # Reduce the size of the wet_cells_map (originally a tuple of Int64)
    N = maximum(size(grid))
    Type = N > MAXUInt8 ? (N > MAXUInt16 ? (N > MAXUInt32 ? UInt64 : UInt32) : UInt16) : UInt8
    smaller_indices = getproperty.(full_indices, Ref(:I)) .|> Tuple{Type, Type, Type}
    
    return smaller_indices
end