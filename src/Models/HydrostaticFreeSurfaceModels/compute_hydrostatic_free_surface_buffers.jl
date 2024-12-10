import Oceananigans.Models: compute_buffer_tendencies!

using Oceananigans.Grids: halo_size
using Oceananigans.DistributedComputations: DistributedActiveCellsIBG
using Oceananigans.ImmersedBoundaries: retrieve_interior_active_cells_map
using Oceananigans.Models.NonhydrostaticModels: buffer_tendency_kernel_parameters,
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

    params3D = buffer_surface_kernel_parameters(grid, arch)
    params3D = buffer_tendency_kernel_parameters(grid, model.closure, arch)

    # We need new values for `w`, `p` and `κ`
    compute_auxiliaries!(model, grid, arch; params2D, params3D, active_cells_map=nothing)

    kernel_parameters = buffer_tendency_kernel_parameters(grid, arch)
    compute_hydrostatic_free_surface_tendency_contributions!(model, params3D)
    return nothing
end

function compute_buffer_tendency_contributions!(grid::DistributedActiveCellsIBG, arch, model)
    maps = grid.interior_active_cells

    # We do not need 3D parameters, they are in the active map
    params2D = buffer_surface_kernel_parameters(grid, arch)

    for (idx, name) in enumerate((:west_halo_dependent_cells, 
                                  :east_halo_dependent_cells,
                                  :south_halo_dependent_cells,
                                  :north_halo_dependent_cells))

        map = maps[name]
        
        # If there exists a buffer map, then we compute the buffer contributions. If not, the 
        # buffer contributions have already been calculated. We exclude the interior because it has
        # already been calculated
        compute_buffer = !isnothing(map) 

        if compute_buffer
            active_cells_map = retrieve_interior_active_cells_map(grid, Val(name))
            compute_auxiliaries!(model; params2D=params2D[idx], params3D=:xyz, active_cells_map)

            compute_hydrostatic_free_surface_tendency_contributions!(model, :xyz; active_cells_map)
        end
    end

    return nothing
end

# w needs computing in the range - H + 1 : 0 and N - 1 : N + H - 1
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

