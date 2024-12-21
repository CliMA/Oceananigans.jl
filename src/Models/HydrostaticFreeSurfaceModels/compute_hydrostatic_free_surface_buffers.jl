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

    # parameters for communicating North / South / East / West side
    compute_buffer_tendency_contributions!(grid, arch, model)

    return nothing
end

function compute_buffer_tendency_contributions!(grid, arch, model) 
    kernel_parameters = buffer_tendency_kernel_parameters(grid, arch)

    w_parameters = buffer_surface_kernel_parameters(grid, arch)
    p_parameters = buffer_surface_kernel_parameters(grid, arch)
    κ_parameters = kernel_parameters

    # We need new values for `w`, `p` and `κ`
    compute_auxiliaries!(model; w_parameters, p_parameters, κ_parameters)

    compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters)
    return nothing
end

function compute_buffer_tendency_contributions!(grid::DistributedActiveCellsIBG, arch, model)
    maps = grid.interior_active_cells
    params2D = buffer_surface_kernel_parameters(grid, arch)

    idx = 1
    for name in (:west_halo_dependent_cells, 
                 :east_halo_dependent_cells, 
                 :south_halo_dependent_cells, 
                 :north_halo_dependent_cells)
        
        active_cells_map = @inbounds maps[name]
        
        # If the map == nothing there is no need to recompute the direction
        # (the boundary is not a processor boundary but a physical boundary)
        if !isnothing(map) 
            # We need new values for `w`, `p` and `κ`
            compute_auxiliaries!(model; w_parameters=params2D[idx], 
                                        p_parameters=params2D[idx], 
                                        κ_parameters=:xyz,  # Do not actually matter here, we use the active map
                                        active_cells_map)
            
            compute_hydrostatic_free_surface_tendency_contributions!(model, :xyz; active_cells_map)
            
            idx += 1
        end
    end

    return nothing
end

# Parameters for quantities that need computing in the range 
# - H + 1 : 0 and N - 1 : N + H - 1
function buffer_surface_kernel_parameters(grid, arch)
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

