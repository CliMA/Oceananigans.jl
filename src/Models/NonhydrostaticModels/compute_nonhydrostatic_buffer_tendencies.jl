import Oceananigans.Models: compute_buffer_tendencies!

using Oceananigans.TurbulenceClosures: required_halo_size_x, required_halo_size_y
using Oceananigans.Grids: XFlatGrid, YFlatGrid, AbstractGrid

# TODO: the code in this file is difficult to understand.
# Rewriting it may be helpful.

# We assume here that top/bottom BC are always synched (no partitioning in z)
function compute_buffer_tendencies!(model::NonhydrostaticModel)
    grid = model.grid
    arch = architecture(grid)

    p_parameters = buffer_w_kernel_parameters(grid, arch)
    κ_parameters = buffer_κ_kernel_parameters(grid, model.closure, arch)

    # We need new values for `p` and `κ`
    compute_auxiliaries!(model; p_parameters, κ_parameters)

    # parameters for communicating North / South / East / West side
    kernel_parameters = buffer_tendency_kernel_parameters(grid, arch)
    compute_interior_tendency_contributions!(model, kernel_parameters)

    return nothing
end

exclude_flat(range, T) = range
exclude_flat(range, ::Type{Flat}) = 1:1

# tendencies need computing in the range 1 : H and N - H + 1 : N
function buffer_tendency_kernel_parameters(grid::AbstractGrid{<:Any, TX, TY, TZ}, arch) where {TX, TY, TZ}
    Nx, Ny, Nz = size(grid)
    Hx, Hy, _  = halo_size(grid)

    param_west  = (1:Hx,       exclude_flat(1:Ny, TY), exclude_flat(1:Nz, TZ))
    param_east  = (Nx-Hx+1:Nx, exclude_flat(1:Ny, TY), exclude_flat(1:Nz, TZ))
    param_south = (exclude_flat(1:Nx, TX), 1:Hy,       exclude_flat(1:Nz, TZ))
    param_north = (exclude_flat(1:Nx, TX), Ny-Hy+1:Ny, exclude_flat(1:Nz, TZ))

    params = (param_west, param_east, param_south, param_north)
    return buffer_parameters(params, grid, arch)
end

# w needs computing in the range - H + 1 : 0 and N - 1 : N + H - 1
function buffer_w_kernel_parameters(grid::AbstractGrid{<:Any, TX, TY, TZ}, arch) where {TX, TY, TZ}
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    # Offsets in tangential direction are == -1 to
    # cover the required corners
    param_west  = (-Hx+2:1,    exclude_flat(0:Ny+1, TY))
    param_east  = (Nx:Nx+Hx-1, exclude_flat(0:Ny+1, TY))
    param_south = (exclude_flat(0:Nx+1, TX),    -Hy+2:1)
    param_north = (exclude_flat(0:Nx+1, TX), Ny:Ny+Hy-1)

    params = (param_west, param_east, param_south, param_north)

    return buffer_parameters(params, grid, arch)
end

# diffusivities need recomputing in the range 0 : B and N - B + 1 : N + 1
function buffer_κ_kernel_parameters(grid::AbstractGrid{<:Any, TX, TY, TZ}, closure, arch) where {TX, TY, TZ}
    Nx, Ny, Nz = size(grid)

    Bx = required_halo_size_x(closure)
    By = required_halo_size_y(closure)

    param_west  = (0:Bx,         exclude_flat(1:Ny, TY), exclude_flat(1:Nz, TZ))
    param_east  = (Nx-Bx+1:Nx+1, exclude_flat(1:Ny, TY), exclude_flat(1:Nz, TZ))
    param_south = (exclude_flat(1:Nx, TX), 0:By,         exclude_flat(1:Nz, TZ))
    param_north = (exclude_flat(1:Nx, TX), Ny-By+1:Ny+1, exclude_flat(1:Nz, TZ))

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

