using Oceananigans.ImmersedBoundaries
using Oceananigans.ImmersedBoundaries:
    AbstractGridFittedBottom,
    GridFittedVerticalBoundaries,
    PartialCellVerticalBoundaries,
    GridFittedBoundary,
    bottom_height_interior,
    compute_mask,
    has_active_cells_map,
    has_active_z_columns,
    active_cells_maps

using Oceananigans.Grids: CommunicatingSide, Sides

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

# The same immersed boundary with its heights mapped by `f`
map_heights(f, ib::GridFittedVerticalBoundaries)  = GridFittedVerticalBoundaries(f(ib.bottom_height), f(ib.top_height), ib.immersed_condition)
map_heights(f, ib::PartialCellVerticalBoundaries) = PartialCellVerticalBoundaries(f(ib.bottom_height), f(ib.top_height), ib.minimum_fractional_cell_height)

function reconstruct_global_immersed_boundary(ib::AbstractGridFittedBottom, arch, grid)
    Nx, Ny, _ = size(grid)
    return map_heights(height -> global_height_array(height, arch, (Nx, Ny, 1)), ib)
end

global_height_array(::Nothing, arch, global_size) = nothing
global_height_array(height, arch, global_size) = construct_global_array(bottom_height_interior(height), arch, global_size)

function reconstruct_global_immersed_boundary(ib::GridFittedBoundary, arch, grid)
    global_mask = construct_global_array(ib.mask, arch, size(grid))
    return GridFittedBoundary(global_mask)
end

function with_halo(new_halo, grid::DistributedImmersedBoundaryGrid)
    active_cells_map      = has_active_cells_map(grid)
    active_z_columns      = has_active_z_columns(grid)
    immersed_boundary     = grid.immersed_boundary
    underlying_grid       = grid.underlying_grid
    new_underlying_grid   = with_halo(new_halo, underlying_grid)
    new_immersed_boundary = resize_immersed_boundary(immersed_boundary, new_underlying_grid)
    return ImmersedBoundaryGrid(new_underlying_grid, new_immersed_boundary; active_cells_map, active_z_columns)
end

function scatter_local_grids(global_grid::ImmersedBoundaryGrid, arch::Distributed, local_size)
    ib = global_grid.immersed_boundary
    ug = global_grid.underlying_grid
    active_cells_map = has_active_cells_map(global_grid)
    active_z_columns = has_active_z_columns(global_grid)

    local_ug = scatter_local_grids(ug, arch, local_size)

    nx, ny, _ = local_size
    local_ib = map_heights(height -> local_height_array(height, arch, (nx, ny, 1)), ib)

    return ImmersedBoundaryGrid(local_ug, local_ib; active_cells_map, active_z_columns)
end

local_height_array(::Nothing, arch, local_size) = nothing
local_height_array(height, arch, local_size) = partition(bottom_height_interior(height), arch, local_size)

"""
    function resize_immersed_boundary!(ib, grid)

If the immersed condition is an `OffsetArray`, resize it to match
the total size of `grid`
"""
resize_immersed_boundary(ib::GridFittedBoundary, grid) = ib

function resize_immersed_boundary(ib::GridFittedBoundary{<:OffsetArray}, grid)

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    mask_size = (Nx, Ny, Nz) .+ 2 .* (Hx, Hy, Hz)

    # Check that the size of a bottom field are
    # consistent with the size of the grid
    if any(size(ib.mask) .!= mask_size)
        @warn "Resizing the mask to match the grids' halos"
        mask = compute_mask(grid, ib)
        return GridFittedBoundary(mask)
    end

    return ib
end

function resize_immersed_boundary(ib::AbstractGridFittedBottom, grid)

    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    height_size = (Nx, Ny, 1) .+ 2 .* (Hx, Hy, 0)

    # Check that the size of the bottom height is consistent with the grid's halos
    if wrong_height_size(ib.bottom_height, height_size) || wrong_height_size(ib.top_height, height_size)
        @warn "Resizing the bottom height to match the grid's halos"
        return map_heights(height -> resize_height(height, grid), ib)
    end

    return ib
end

wrong_height_size(::Nothing, height_size) = false
wrong_height_size(height, height_size) = any(size(height) .!= height_size)

resize_height(::Nothing, grid) = nothing

function resize_height(height, grid)
    Nx, Ny, _ = size(grid)
    height_field = Field{Center, Center, Nothing}(grid)
    cpu_height   = on_architecture(CPU(), height)[1:Nx, 1:Ny, :]
    set!(height_field, cpu_height)
    fill_halo_regions!(height_field)
    return height_field.data
end

# In case of an asynchronous `DistributedGrid` we want to have different maps depending on the partitioning of the domain:
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

    one_sided_x = (Tx <: Sides{<:CommunicatingSide}) != (Tx <: Sides{<:Any, <:CommunicatingSide})
    one_sided_y = (Ty <: Sides{<:CommunicatingSide}) != (Ty <: Sides{<:Any, <:CommunicatingSide})

    nx = Rx == 1 ? Nx : (one_sided_x ? Nx - Hx : Nx - 2Hx)
    ny = Ry == 1 ? Ny : (one_sided_y ? Ny - Hy : Ny - 2Hy)

    ox = Rx == 1 || !(Tx <: Sides{<:CommunicatingSide}) ? 0 : Hx
    oy = Ry == 1 || !(Ty <: Sides{<:CommunicatingSide}) ? 0 : Hy

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
