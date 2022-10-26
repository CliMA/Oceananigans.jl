using Oceananigans
import Oceananigans.Utils: only_active_cells_in_worksize, calc_tendency_index
using KernelAbstractions
using KernelAbstractions: @index

only_active_cells_in_worksize(size, grid::IBG) = min(length(grid.wet_cells_map), 512), length(grid.wet_cells_map)

@inline calc_tendency_index(idx, i, j, k, grid::IBG) = grid.wet_cells_map[idx].I

function ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib) where {TX, TY, TZ} 

    # Create the cells map on the CPU, then switch it to the GPU
    wet_cells_map    = create_cells_map(grid, ib)
    wet_cells_map    = arch_array(architecture(grid), wet_cells_map)

    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib, wet_cells_map)
end

wet_cell(i, j, k, grid, ib) = !immersed_cell(i, j, k, grid, ib)

function compute_wet_cells(grid, ib)
    is_immersed_operation = KernelFunctionOperation{Center, Center, Center}(wet_cell, grid; computed_dependencies = (ib, ))
    wet_cells_field = Field{Center, Center, Center}(grid, Bool)
    set!(wet_cells_field, is_immersed_operation)
    return wet_cells_field
end

using Oceananigans.Fields: conditional_length

function create_cells_map(grid, ib)
    wet_cells_field = compute_wet_cells(grid, ib)
    return findall(interior(wet_cells_field))
end
