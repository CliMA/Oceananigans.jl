import Oceananigans.Models: compute_buffer_tendencies!

using Oceananigans.TurbulenceClosures: required_halo_size_x, required_halo_size_y
using Oceananigans.Grids: XFlatGrid, YFlatGrid

# TODO: the code in this file is difficult to understand.
# Rewriting it may be helpful.

# We assume here that top/bottom BC are always synched (no partitioning in z)
function compute_buffer_tendencies!(model::NonhydrostaticModel)
    grid = model.grid
    arch = architecture(grid)

    params2D = buffer_surface_kernel_parameters(grid, arch)
    params3D = buffer_tendency_kernel_parameters(grid, arch)
    
    # We need new values for `p` and `κ`
    compute_auxiliaries!(model, grid, arch; params2D, params3D)

    # parameters for communicating North / South / East / West side
    compute_interior_tendency_contributions!(model, kernel_parameters)

    return nothing
end

# surface needs computing in the range - H + 1 : 0 and N - 1 : N + H - 1
function buffer_surface_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    # Offsets in tangential direction are == -1 to
    # cover the required corners
    param_west  = (-Hx+2:1,    0:Ny+1)
    param_east  = (Nx:Nx+Hx-1, 0:Ny+1)
    param_south = (0:Nx+1,     -Hy+2:1)
    param_north = (0:Nx+1,     Ny:Ny+Hy-1)

    params = (param_west, param_east, param_south, param_north)

    return buffer_parameters(params, grid, arch)
end

# tendencies need computing in the range 1 : H and N - H + 1 : N 
function buffer_tendency_kernel_parameters(grid, arch)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, _  = halo_size(grid)

    param_west  = (0:Hx,         1:Ny,         1:Nz)
    param_east  = (Nx-Hx+1:Nx+1, 1:Ny,         1:Nz)
    param_south = (1:Nx,         0:Hy,         1:Nz)
    param_north = (1:Nx,         Ny-Hy+1:Ny+1, 1:Nz)

    params = (param_west, param_east, param_south, param_north)
    return buffer_parameters(params, grid, arch)
end

# Recompute only on communicating sides 
function buffer_parameters(parameters, grid, arch) 
    Rx, Ry, _ = arch.ranks
    Tx, Ty, _ = topology(grid)

    include_west  = !isa(grid, XFlatGrid) && (Rx != 1) && !(Tx == RightConnected)
    include_east  = !isa(grid, XFlatGrid) && (Rx != 1) && !(Tx == LeftConnected)
    include_south = !isa(grid, YFlatGrid) && (Ry != 1) && !(Ty == RightConnected)
    include_north = !isa(grid, YFlatGrid) && (Ry != 1) && !(Ty == LeftConnected)

    include_side = (include_west, include_east, include_south, include_north)

    return Tuple(KernelParameters(parameters[i]...) for i in findall(include_side))
end

