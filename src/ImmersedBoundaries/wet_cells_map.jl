import Oceananigans.Utils: remove_immersed_boundaries_from_worksize, calc_tendency_index

remove_immersed_boundary_from_worksize(size, grid::IBG) = (length(grid.wet_cells_map), 1, 1)

@inline calc_tendency_index(i, j, k, grid::IBG) = grid.wet_cells_map[i].I

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
    grid_cpu = on_architecture(Oceananigans.Architectures.CPU(), grid)

    Nx, Ny, Nz = N = size(grid_cpu)
    
    wet_cells_field = compute_wet_cells(grid, ib)

    return findall(interior(wet_cells_field))
end
