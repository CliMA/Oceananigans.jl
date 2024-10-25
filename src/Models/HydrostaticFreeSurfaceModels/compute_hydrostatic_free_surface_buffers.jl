import Oceananigans.Models: compute_buffer_tendencies!

using Oceananigans.Grids: halo_size
using Oceananigans.DistributedComputations: DistributedActiveCellsIBG
using Oceananigans.ImmersedBoundaries: retrieve_interior_active_cells_map
using Oceananigans.Models.NonhydrostaticModels: buffer_tendency_kernel_parameters,
                                                buffer_p_kernel_parameters, 
                                                buffer_κ_kernel_parameters,
                                                buffer_parameters

# We assume here that top/bottom BC are always synchronized (no partitioning in z)
function compute_buffer_tendencies!(model::HydrostaticFreeSurfaceModel)
    grid = model.grid
    arch = architecture(grid)

    w_parameters = buffer_w_kernel_parameters(grid, arch)
    p_parameters = buffer_p_kernel_parameters(grid, arch)
    κ_parameters = buffer_κ_kernel_parameters(grid, model.closure, arch)

    # We need new values for `w`, `p` and `κ`
    compute_auxiliaries!(model; w_parameters, p_parameters, κ_parameters)

    # parameters for communicating North / South / East / West side
    compute_buffer_tendency_contributions!(grid, arch, model)

    return nothing
end

function compute_buffer_tendency_contributions!(grid, arch, model) 
    kernel_parameters = buffer_tendency_kernel_parameters(grid, arch)
    compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters)
    return nothing
end

function compute_buffer_tendency_contributions!(grid::DistributedActiveCellsIBG, arch, model)
    maps = grid.interior_active_cells
    
    for (name, map) in zip(keys(maps), maps)
        
        # If there exists a buffer map, then we compute the buffer contributions. If not, the 
        # buffer contributions have already been calculated. We exclude the interior because it has
        # already been calculated
        compute_buffer = (name != :interior) && !isnothing(map) 

        if compute_buffer
            active_cells_map = retrieve_interior_active_cells_map(grid, Val(name))
            compute_hydrostatic_free_surface_tendency_contributions!(model, :xyz; active_cells_map)
        end
    end

    return nothing
end

# w needs computing in the range - H + 1 : 0 and N - 1 : N + H - 1
function buffer_w_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    Sx  = (Hx, Ny+2) 
    Sy  = (Nx+2, Hy)
             
    # Offsets in tangential direction are == -1 to
    # cover the required corners
    param_west  = (-Hx+2:1,    0:Ny+1)
    param_east  = (Nx:Nx+Hx-1, 0:Ny+1)
    param_south = (0:Nx+1,     -Hy+2:1)
    param_north = (0:Nx+1,     Ny:Ny+Hy-1)

    params = (param_west, param_east, param_south, param_north)

    return buffer_parameters(params, grid, arch)
end

