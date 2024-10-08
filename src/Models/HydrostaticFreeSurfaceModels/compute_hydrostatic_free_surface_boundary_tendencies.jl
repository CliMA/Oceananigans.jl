import Oceananigans.Models: compute_boundary_tendencies!
import Oceananigans.Models: compute_boundary_tendencies!

using Oceananigans.Grids: halo_size
using Oceananigans.ImmersedBoundaries: retrieve_interior_active_cells_map, DistributedActiveCellsIBG
using Oceananigans.Models.NonhydrostaticModels: boundary_tendency_kernel_parameters,
                                                boundary_p_kernel_parameters, 
                                                boundary_κ_kernel_parameters,
                                                boundary_parameters

# We assume here that top/bottom BC are always synchronized (no partitioning in z)
function compute_boundary_tendencies!(model::HydrostaticFreeSurfaceModel)
    grid = model.grid
    arch = architecture(grid)

    w_parameters = boundary_w_kernel_parameters(grid, arch)
    p_parameters = boundary_p_kernel_parameters(grid, arch)
    κ_parameters = boundary_κ_kernel_parameters(grid, model.closure, arch)

    # Compute new values for `w`, `p` and `κ` on the perifery
    compute_auxiliaries!(model; w_parameters, p_parameters, κ_parameters)

    # parameters for communicating North / South / East / West side
    compute_boundary_tendency_contributions!(grid, arch, model)

    return nothing
end

function compute_boundary_tendency_contributions!(grid, arch, model) 
    kernel_parameters = boundary_tendency_kernel_parameters(grid, arch)
    compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters)
    return nothing
end

function compute_boundary_tendency_contributions!(grid::DistributedActiveCellsIBG, arch, model)
    maps = grid.interior_active_cells
    
    for (name, map) in zip(keys(maps), maps)
        
        # If there exists a boundary map, then we compute the boundary contributions. If not, the 
        # boundary contributions have already been calculated. We exclude the interior because it has
        # already been calculated
        compute_boundary = (name != :interior) && !isnothing(map) 

        if compute_boundary
            active_cells_map = retrieve_interior_active_cells_map(grid, Val(name))
            compute_hydrostatic_free_surface_tendency_contributions!(model, tuple(:xyz); active_cells_map)
        end
    end

    return nothing
end

# w needs computing in the range - H + 1 : 0 and N - 1 : N + H - 1
function boundary_w_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    # Offsets in tangential direction are == -1 to
    # cover the required corners
    param_west  = (-Hx+1:0,      0:Ny+1,       1:Nz)
    param_east  = (Nx+1:Nx+Hx-1, 0:Ny+1,       1:Nz)
    param_north = (0:Nx+1,       -Hy+1:0,      1:Nz)
    param_south = (0:Nx+1,       Ny+1:Ny+Hy-1, 1:Nz)

    params = (param_west, param_east, param_north, param_south)

    return boundary_parameters(params, grid, arch)
end

