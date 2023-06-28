import Oceananigans.Distributed: compute_boundary_tendencies!
using Oceananigans.Utils: worktuple, offsets
using Oceananigans.TurbulenceClosures: required_halo_size
using Oceananigans.Models.NonhydrostaticModels: boundary_tendency_kernel_parameters,
                                                boundary_p_kernel_parameters, 
                                                boundary_κ_kernel_parameters,
                                                boundary_parameters

import Oceananigans.Models.NonhydrostaticModels: compute_boundary_tendencies!

                                
# We assume here that top/bottom BC are always synched (no partitioning in z)
function compute_boundary_tendencies!(model::HydrostaticFreeSurfaceModel)
    grid = model.grid
    arch = architecture(grid)

    # We need new values for `w`, `p` and `κ`
    recompute_auxiliaries!(model, grid, arch)

    # parameters for communicating North / South / East / West side
    kernel_parameters = boundary_tendency_kernel_parameters(grid, arch)
    calculate_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters)

    return nothing
end

function recompute_auxiliaries!(model::HydrostaticFreeSurfaceModel, grid, arch)
    
    w_kernel_parameters = boundary_w_kernel_parameters(grid, arch)
    p_kernel_parameters = boundary_p_kernel_parameters(grid, arch)
    κ_kernel_parameters = boundary_κ_kernel_parameters(grid, model.closure, arch)

    for (wpar, ppar, κpar) in zip(w_kernel_parameters, p_kernel_parameters, κ_kernel_parameters)
        compute_w_from_continuity!(model.velocities, arch, grid; parameters = wpar)
        update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers; parameters = ppar)
        calculate_diffusivities!(model.diffusivity_fields, model.closure, model; parameters = κpar)
    end
end

# w needs computing in the range - H + 1 : 0 and N - 1 : N + H - 1
function boundary_w_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    Sx  = (Hx, Ny)
    Sy  = (Nx, Hy)
             
    Oxᴸ = (-Hx+1, 0)
    Oyᴸ = (0, -Hy+1)
    Oxᴿ = (Nx-1,  0)
    Oyᴿ = (0,  Ny-1)

    sizes = (Sx,  Sy,  Sx,  Sy)
    offs  = (Oxᴸ, Oyᴸ, Oxᴿ, Oyᴿ)
        
    return boundary_parameters(sizes, offs, grid, arch)
end

