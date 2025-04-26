using Oceananigans.Utils: getnamewrapper
using Oceananigans.ImmersedBoundaries
using Oceananigans.ImmersedBoundaries:
    AbstractGridFittedBottom,
    GridFittedBottom,
    GridFittedBoundary,
    compute_mask,
    CellMaps,
    has_active_cells_map,
    has_active_z_columns,
    serially_build_active_cells_map,
    compute_mask

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
    global_ib = getnamewrapper(local_ib)(construct_global_array(local_ib.bottom_height, arch, size(grid)))
    return ImmersedBoundaryGrid(global_ug, global_ib; active_cells_map, active_z_columns)
end

function with_halo(new_halo, grid::DistributedImmersedBoundaryGrid)
    active_cells_map      = has_active_cells_map(grid)
    active_z_columns      = has_active_z_columns(grid)
    immersed_boundary     = grid.immersed_boundary
    underlying_grid       = grid.underlying_grid
    new_underlying_grid   = with_halo(new_halo, underlying_grid)
    new_immersed_boundary = resize_immersed_boundary(immersed_boundary, new_underlying_grid)
    return ImmersedBoundaryGrid(new_underlying_grid, new_immersed_boundary;
                                active_cells_map, active_z_columns)
end

function scatter_local_grids(global_grid::ImmersedBoundaryGrid, arch::Distributed, local_size)
    ib = global_grid.immersed_boundary
    ug = global_grid.underlying_grid
    active_cells_map = has_active_cells_map(global_grid)
    active_z_columns = has_active_z_columns(global_grid)

    local_ug = scatter_local_grids(ug, arch, local_size)

    # Kinda hacky
    local_bottom_height = partition(ib.bottom_height, arch, local_size)
    ImmersedBoundaryConstructor = getnamewrapper(ib)
    local_ib = ImmersedBoundaryConstructor(local_bottom_height)

    return ImmersedBoundaryGrid(local_ug, local_ib; active_cells_map, active_z_columns)
end

"""
    function resize_immersed_boundary!(ib, grid)

If the immersed condition is an `OffsetArray`, resize it to match
the total size of `grid`
"""
resize_immersed_boundary(ib::AbstractGridFittedBottom, grid) = ib
resize_immersed_boundary(ib::GridFittedBoundary, grid) = ib

function resize_immersed_boundary(ib::GridFittedBoundary{<:OffsetArray}, grid)

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Nz = halo_size(grid)

    mask_size = (Nx, Ny, Nz) .+ 2 .* (Hx, Hy, Hz)

    # Check that the size of a bottom field are
    # consistent with the size of the grid
    if any(size(ib.mask) .!= mask_size)
        @warn "Resizing the mask to match the grids' halos"
        mask = compute_mask(grid, ib)
        return getnamewrapper(ib)(mask)
    end

    return ib
end

function resize_immersed_boundary(ib::AbstractGridFittedBottom{<:OffsetArray}, grid)

    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    bottom_heigth_size = (Nx, Ny) .+ 2 .* (Hx, Hy)

    # Check that the size of a bottom field are
    # consistent with the size of the grid
    if any(size(ib.bottom_height) .!= bottom_heigth_size)
        @warn "Resizing the bottom field to match the grids' halos"
        bottom_field = Field((Center, Center, Nothing), grid)
        cpu_bottom   = on_architecture(CPU(), ib.bottom_height)[1:Nx, 1:Ny]
        set!(bottom_field, cpu_bottom)
        fill_halo_regions!(bottom_field)
        offset_bottom_array = dropdims(bottom_field.data, dims=3)

        return getnamewrapper(ib)(offset_bottom_array)
    end

    return ib
end

# In case of a `DistributedGrid` we want to have different maps depending on the partitioning of the domain:
#
# If we partition the domain in the x-direction, we typically want to have the option to split three-dimensional
# kernels in a `halo-independent` part in the range Hx+1:Nx-Hx, 1:Ny, 1:Nz and two `halo-dependent` computations:
# a west one spanning 1:Hx, 1:Ny, 1:Nz and an east one spanning Nx-Hx+1:Nx, 1:Ny, 1:Nz.
# For this reason we need three different maps, one containing the `halo_independent` active region, a `west` map and an `east` map.
# For the same reason we need to construct `south` and `north` maps if we partition the domain in the y-direction.
# Therefore, the `interior_active_cells` in this case is a `NamedTuple` containing 5 elements.
# Note that boundary-adjacent maps corresponding to non-partitioned directions are set to `nothing`
function build_active_cells_map(grid::DistributedGrid, ib)

    arch = architecture(grid)

    # If we using a synchronized architecture, nothing
    # changes with serial execution.
    if !(arch isa AsynchronousDistributed)
        return serially_build_active_cells_map(grid, ib; parameters=:xyz)
    end

    Rx, Ry, _  = arch.ranks
    Tx, Ty, _  = topology(grid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, _  = halo_size(grid)

    west_boundary  = (1:Hx,       1:Ny, 1:Nz)
    east_boundary  = (Nx-Hx+1:Nx, 1:Ny, 1:Nz)
    south_boundary = (1:Nx, 1:Hy,       1:Nz)
    north_boundary = (1:Nx, Ny-Hy+1:Ny, 1:Nz)

    include_west  = !isa(grid, XFlatGrid) && (Rx != 1) && !(Tx == RightConnected)
    include_east  = !isa(grid, XFlatGrid) && (Rx != 1) && !(Tx == LeftConnected)
    include_south = !isa(grid, YFlatGrid) && (Ry != 1) && !(Ty == RightConnected)
    include_north = !isa(grid, YFlatGrid) && (Ry != 1) && !(Ty == LeftConnected)

    west_halo_dependent_cells  = serially_build_active_cells_map(grid, ib; parameters = KernelParameters(west_boundary...))
    east_halo_dependent_cells  = serially_build_active_cells_map(grid, ib; parameters = KernelParameters(east_boundary...))
    south_halo_dependent_cells = serially_build_active_cells_map(grid, ib; parameters = KernelParameters(south_boundary...))
    north_halo_dependent_cells = serially_build_active_cells_map(grid, ib; parameters = KernelParameters(north_boundary...))

    west_halo_dependent_cells  = ifelse(include_west,  west_halo_dependent_cells,  nothing)
    east_halo_dependent_cells  = ifelse(include_east,  east_halo_dependent_cells,  nothing)
    south_halo_dependent_cells = ifelse(include_south, south_halo_dependent_cells, nothing)
    north_halo_dependent_cells = ifelse(include_north, north_halo_dependent_cells, nothing)

    nx = Rx == 1 ? Nx : (Tx == RightConnected || Tx == LeftConnected ? Nx - Hx : Nx - 2Hx)
    ny = Ry == 1 ? Ny : (Ty == RightConnected || Ty == LeftConnected ? Ny - Hy : Ny - 2Hy)

    ox = Rx == 1 || Tx == RightConnected ? 0 : Hx
    oy = Ry == 1 || Ty == RightConnected ? 0 : Hy

    halo_independent_cells = serially_build_active_cells_map(grid, ib; parameters = KernelParameters((nx, ny, Nz), (ox, oy, 0)))

    return (; halo_independent_cells,
              west_halo_dependent_cells,
              east_halo_dependent_cells,
              south_halo_dependent_cells,
              north_halo_dependent_cells)
end
