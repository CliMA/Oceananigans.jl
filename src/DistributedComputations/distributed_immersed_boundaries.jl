using Oceananigans.Utils: getnamewrapper
using Oceananigans.ImmersedBoundaries
using Oceananigans.ImmersedBoundaries:
    GridFittedBottom,
    PartialCellBottom,
    GridFittedBoundary,
    bottom_height_interior,
    has_active_cells_map,
    has_active_z_columns,
    active_cells_maps

import Oceananigans.ImmersedBoundaries: build_active_cells_map

# For the moment we extend distributed in the `ImmersedBoundaryGrids` module.
# When we fix the immersed boundary module to remove all the `TurbulenceClosure` stuff
# we can move this file back to `DistributedComputations` if we want `ImmersedBoundaries`
# to take precedence
const DistributedImmersedBoundaryGrid = ImmersedBoundaryGrid{FT, TX, TY, TZ,
                                                             <:DistributedGrid, I, M, S,
                                                             <:Distributed} where {FT, TX, TY, TZ, I, M, S}

function reconstruct_global_grid(grid::ImmersedBoundaryGrid)
    active_cells_map = has_active_cells_map(grid)
    active_z_columns = has_active_z_columns(grid)
    arch      = grid.architecture
    local_ib  = grid.immersed_boundary
    global_ug = reconstruct_global_grid(grid.underlying_grid)
    global_ib = reconstruct_global_immersed_boundary(local_ib, arch, grid)
    return ImmersedBoundaryGrid(global_ug, global_ib; active_cells_map, active_z_columns)
end

function reconstruct_global_immersed_boundary(ib::GridFittedBottom, arch, grid)
    Nx, Ny, _ = size(grid)
    bottom_interior = bottom_height_interior(ib.bottom_height)
    global_bottom_height = construct_global_array(bottom_interior, arch, (Nx, Ny, 1))
    return GridFittedBottom(global_bottom_height, ib.immersed_condition)
end

function reconstruct_global_immersed_boundary(ib::PartialCellBottom, arch, grid)
    Nx, Ny, _ = size(grid)
    bottom_interior = bottom_height_interior(ib.bottom_height)
    global_bottom_height = construct_global_array(bottom_interior, arch, (Nx, Ny, 1))
    return PartialCellBottom(global_bottom_height, ib.minimum_fractional_cell_height)
end

function reconstruct_global_immersed_boundary(ib::GridFittedBoundary, arch, grid)
    global_mask = construct_global_array(ib.mask, arch, size(grid))
    return GridFittedBoundary(global_mask)
end

# The active cells maps are rebuilt because their partition depends on the halo.
function with_halo(new_halo, grid::DistributedImmersedBoundaryGrid)
    active_cells_map    = has_active_cells_map(grid)
    active_z_columns    = has_active_z_columns(grid)
    new_underlying_grid = with_halo(new_halo, grid.underlying_grid)
    return ImmersedBoundaryGrid(new_underlying_grid, grid.immersed_boundary; active_cells_map, active_z_columns)
end

function scatter_local_grids(global_grid::ImmersedBoundaryGrid, arch::Distributed, local_size)
    ib = global_grid.immersed_boundary
    ug = global_grid.underlying_grid
    active_cells_map = has_active_cells_map(global_grid)
    active_z_columns = has_active_z_columns(global_grid)

    local_ug = scatter_local_grids(ug, arch, local_size)

    nx, ny, _ = local_size
    bottom_interior = bottom_height_interior(ib.bottom_height)
    local_bottom_height = partition(bottom_interior, arch, (nx, ny, 1))
    ImmersedBoundaryConstructor = getnamewrapper(ib)
    local_ib = ImmersedBoundaryConstructor(local_bottom_height)

    return ImmersedBoundaryGrid(local_ug, local_ib; active_cells_map, active_z_columns)
end

# In case of a `DistributedGrid` we want to have different maps depending on the partitioning of the domain:
#
# If we partition the domain in the x-direction, we typically want to have the option to split three-dimensional
# kernels in a `halo-independent` part in the range Hx+1:Nx-Hx, 1:Ny, 1:Nz and two `halo-dependent` computations:
# a west one spanning 1:Hx, 1:Ny, 1:Nz and an east one spanning Nx-Hx+1:Nx, 1:Ny, 1:Nz.
# For this reason we need three different maps, one containing the `halo_independent` active region, a `west` map and an `east` map.
# For the same reason we need to construct `south` and `north` maps if we partition the domain in the y-direction.
# The south and north maps span only the x-range of the `halo_independent` region, so that the five maps are disjoint.
# Therefore, the `interior_active_cells` in this case is a `NamedTuple` containing these 5 elements and their union.
# Note that boundary-adjacent maps corresponding to non-partitioned directions are set to `nothing`
function build_active_cells_map(grid::AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:AsynchronousDistributed}, ib)
    arch = architecture(grid)
    Rx, Ry, _  = arch.ranks
    Tx, Ty, _  = topology(grid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, _  = halo_size(grid)

    nx = Rx == 1 ? Nx : (Tx == RightConnected || Tx == LeftConnected ? Nx - Hx : Nx - 2Hx)
    ny = Ry == 1 ? Ny : (Ty == RightConnected || Ty == LeftConnected ? Ny - Hy : Ny - 2Hy)

    ox = Rx == 1 || Tx == RightConnected ? 0 : Hx
    oy = Ry == 1 || Ty == RightConnected ? 0 : Hy

    regions = ((ox+1:ox+nx, oy+1:oy+ny, 1:Nz),
               (1:ox,       1:Ny,       1:Nz),
               (ox+nx+1:Nx, 1:Ny,       1:Nz),
               (ox+1:ox+nx, 1:oy,       1:Nz),
               (ox+1:ox+nx, oy+ny+1:Ny, 1:Nz))

    interior, halo_independent_cells, halo_dependent_cells... = active_cells_maps(grid, ib, regions)

    west_halo_dependent_cells, east_halo_dependent_cells, south_halo_dependent_cells, north_halo_dependent_cells = map(cells -> isempty(cells) ? nothing : cells, halo_dependent_cells)

    return (; interior,
              halo_independent_cells,
              west_halo_dependent_cells,
              east_halo_dependent_cells,
              south_halo_dependent_cells,
              north_halo_dependent_cells)
end
