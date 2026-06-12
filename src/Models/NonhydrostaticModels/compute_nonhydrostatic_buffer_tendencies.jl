import Oceananigans.Models: compute_buffer_tendencies!

using Oceananigans.TurbulenceClosures: required_halo_size_x, required_halo_size_y
using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans.Utils: worksize

# TODO: the code in this file is difficult to understand.
# Rewriting it may be helpful.

# We assume here that top/bottom BC are always synched (no partitioning in z)
function compute_buffer_tendencies!(model::NonhydrostaticModel)
    grid = model.grid
    arch = architecture(grid)

    p_parameters = buffer_p_kernel_parameters(grid, arch)
    κ_parameters = buffer_κ_kernel_parameters(grid, model.closure, arch)

    # We need new values for `p` and `κ`
    compute_auxiliaries!(model; p_parameters, κ_parameters)

    # parameters for communicating North / South / East / West side
    kernel_parameters = buffer_tendency_kernel_parameters(grid, arch)
    compute_interior_tendency_contributions!(model, kernel_parameters)

    return nothing
end

# Tendencies need computing in the range 1 : H and W - H + 1 : W where W = worksize.
# These buffer regions complement the halo-independent interior kernel (H+1:W-H).
function buffer_tendency_kernel_parameters(grid, arch)
    Nx, Ny, Nz = size(grid)
    Wx, Wy, _  = worksize(grid)
    Hx, Hy, _  = halo_size(grid)

    param_west  = (1:Hx,       1:Wy,       1:Nz)
    param_east  = (Wx-Hx+1:Wx, 1:Wy,       1:Nz)
    param_south = (1:Wx,       1:Hy,       1:Nz)
    param_north = (1:Wx,       Wy-Hy+1:Wy, 1:Nz)

    params = (param_west, param_east, param_south, param_north)
    return buffer_parameters(params, grid, arch)
end

# p needs computing in the range  0 : 0 and W + 1 : W + 1 where W = worksize
function buffer_p_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)
    Wx, Wy, _ = worksize(grid)

    param_west  = (0:0,       1:Wy)
    param_east  = (Wx+1:Wx+1, 1:Wy)
    param_south = (1:Wx,      0:0)
    param_north = (1:Wx,      Wy+1:Wy+1)

    params = (param_west, param_east, param_south, param_north)
    return buffer_parameters(params, grid, arch)
end

# closure_fields need recomputing in the range 0 : B and W - B + 1 : W + 1 where W = worksize
function buffer_κ_kernel_parameters(grid, closure, arch)
    Nx, Ny, Nz = size(grid)
    Wx, Wy, _  = worksize(grid)

    Bx = required_halo_size_x(closure)
    By = required_halo_size_y(closure)

    param_west  = (0:Bx,         1:Wy,         1:Nz)
    param_east  = (Wx-Bx+1:Wx+1, 1:Wy,         1:Nz)
    param_south = (1:Wx,         0:By,         1:Nz)
    param_north = (1:Wx,         Wy-By+1:Wy+1, 1:Nz)

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
