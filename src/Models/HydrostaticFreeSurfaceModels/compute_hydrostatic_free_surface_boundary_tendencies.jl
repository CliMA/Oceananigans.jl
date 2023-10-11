import Oceananigans.Models: compute_boundary_tendencies!
using Oceananigans.TurbulenceClosures: required_halo_size
using Oceananigans.Models.NonhydrostaticModels: boundary_tendency_kernel_parameters,
                                                boundary_p_kernel_parameters, 
                                                boundary_κ_kernel_parameters,
                                                boundary_parameters

import Oceananigans.Models: compute_boundary_tendencies!

# We assume here that top/bottom BC are always synched (no partitioning in z)
function compute_boundary_tendencies!(model::HydrostaticFreeSurfaceModel)
    grid = model.grid
    arch = architecture(grid)

    w_parameters = boundary_w_kernel_parameters(grid, arch)
    p_parameters = boundary_p_kernel_parameters(grid, arch)
    κ_parameters = boundary_κ_kernel_parameters(grid, model.closure, arch)

    # We need new values for `w`, `p` and `κ`
    compute_auxiliaries!(model; w_parameters, p_parameters, κ_parameters)

    # parameters for communicating North / South / East / West side
    kernel_parameters = boundary_tendency_kernel_parameters(grid, arch)
    compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters)

    return nothing
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

