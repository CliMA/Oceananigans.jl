import Oceananigans.Models: compute_buffer_tendencies!

using Oceananigans.TurbulenceClosures: required_halo_size_x, required_halo_size_y
using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans.DistributedComputations: DistributedActiveInteriorIBG

# We assume here that top/bottom BC are always synched (no partitioning in z)
function compute_buffer_tendencies!(model::NonhydrostaticModel)
    grid = model.grid
    arch = architecture(grid)

    p_parameters = buffer_p_kernel_parameters(grid, arch)
    κ_parameters = buffer_κ_kernel_parameters(grid, model.closure, arch)

    # We need new values for `p` and `κ`
    compute_auxiliaries!(model; p_parameters, κ_parameters)

    # Compute tendencies in buffer regions
    compute_buffer_tendency_contributions!(grid, arch, model)

    return nothing
end

# Default: use KernelParameters ranges for each communicating side
function compute_buffer_tendency_contributions!(grid, arch, model::NonhydrostaticModel)
    kernel_parameters = buffer_tendency_kernel_parameters(grid, arch)
    compute_interior_tendency_contributions!(model, kernel_parameters)
    return nothing
end

# Distributed IBG: iterate over halo-dependent active cells maps
function compute_buffer_tendency_contributions!(grid::DistributedActiveInteriorIBG, arch, model::NonhydrostaticModel)
    maps = grid.interior_active_cells
    for name in (:west_halo_dependent_cells, :east_halo_dependent_cells,
                 :south_halo_dependent_cells, :north_halo_dependent_cells)
        active_cells_map = maps[name]
        if !isnothing(active_cells_map)
            compute_interior_tendency_contributions!(model, nothing; active_cells_map)
        end
    end
    return nothing
end

# tendencies need computing in the range 1 : H and N - H + 1 : N
function buffer_tendency_kernel_parameters(grid, arch)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, _  = halo_size(grid)

    param_west  = (1:Hx,       1:Ny,       1:Nz)
    param_east  = (Nx-Hx+1:Nx, 1:Ny,       1:Nz)
    param_south = (1:Nx,       1:Hy,       1:Nz)
    param_north = (1:Nx,       Ny-Hy+1:Ny, 1:Nz)

    params = (param_west, param_east, param_south, param_north)
    return buffer_parameters(params, grid, arch)
end

# p needs computing in the range  0 : 0 and N + 1 : N + 1
function buffer_p_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)

    param_west  = (0:0,       1:Ny)
    param_east  = (Nx+1:Nx+1, 1:Ny)
    param_south = (1:Nx,      0:0)
    param_north = (1:Nx,      Ny+1:Ny+1)

    params = (param_west, param_east, param_south, param_north)
    return buffer_parameters(params, grid, arch)
end

# diffusivities need recomputing in the range 0 : B and N - B + 1 : N + 1
function buffer_κ_kernel_parameters(grid, closure, arch)
    Nx, Ny, Nz = size(grid)

    Bx = required_halo_size_x(closure)
    By = required_halo_size_y(closure)

    param_west  = (0:Bx,         1:Ny,         1:Nz)
    param_east  = (Nx-Bx+1:Nx+1, 1:Ny,         1:Nz)
    param_south = (1:Nx,         0:By,         1:Nz)
    param_north = (1:Nx,         Ny-By+1:Ny+1, 1:Nz)

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
