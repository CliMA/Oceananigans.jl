import Oceananigans.Models: compute_buffer_tendencies!

using Oceananigans.Grids: halo_size
using Oceananigans.DistributedComputations: Distributed, DistributedGrid
using Oceananigans.ImmersedBoundaries: get_active_cells_map, CellMaps
using Oceananigans.Models.NonhydrostaticModels: buffer_tendency_kernel_parameters,
                                                buffer_p_kernel_parameters,
                                                buffer_κ_kernel_parameters,
                                                buffer_parameters

const DistributedActiveInteriorIBG = ImmersedBoundaryGrid{FT, TX, TY, TZ,
                                                          <:DistributedGrid, I, <:CellMaps, S,
                                                          <:Distributed} where {FT, TX, TY, TZ, I, S}

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

function compute_buffer_tendency_contributions!(grid::DistributedActiveInteriorIBG, arch, model)
    maps = grid.interior_active_cells

    for name in (:west_halo_dependent_cells,
                 :east_halo_dependent_cells,
                 :south_halo_dependent_cells,
                 :north_halo_dependent_cells)

        active_cells_map = @inbounds maps[name]

        # If the map == nothing, we don't need to compute the buffer because
        # the buffer is not adjacent to a processor boundary
        !isnothing(map) && compute_hydrostatic_free_surface_tendency_contributions!(model, :xyz; active_cells_map)
    end

    return nothing
end

# w needs computing in the range - H + 1 : 0 and N - 1 : N + H - 1
function buffer_w_kernel_parameters(grid, arch)
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

