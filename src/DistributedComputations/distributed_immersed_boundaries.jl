using Oceananigans.Utils: getnamewrapper
using Oceananigans.ImmersedBoundaries
using Oceananigans.ImmersedBoundaries: AbstractGridFittedBottom, 
                                       GridFittedBottom, 
                                       GridFittedBoundary, 
                                       compute_mask,
                                       interior_active_indices

import Oceananigans.ImmersedBoundaries: map_interior_active_cells

# For the moment we extend distributed in the `ImmersedBoundaryGrids` module.
# When we fix the immersed boundary module to remove all the `TurbulenceClosure` stuff
# we can move this file back to `DistributedComputations` if we want `ImmersedBoundaries`
# to take precedence
const DistributedImmersedBoundaryGrid = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:DistributedGrid, I, M, <:Distributed} where {FT, TX, TY, TZ, I, M}

function reconstruct_global_grid(grid::ImmersedBoundaryGrid)
    arch      = grid.architecture
    local_ib  = grid.immersed_boundary    
    global_ug = reconstruct_global_grid(grid.underlying_grid)
    global_ib = getnamewrapper(local_ib)(construct_global_array(arch, local_ib.bottom_height, size(grid)))
    return ImmersedBoundaryGrid(global_ug, global_ib)
end

function with_halo(new_halo, grid::DistributedImmersedBoundaryGrid)
    immersed_boundary     = grid.immersed_boundary
    underlying_grid       = grid.underlying_grid
    new_underlying_grid   = with_halo(new_halo, underlying_grid)
    new_immersed_boundary = resize_immersed_boundary(immersed_boundary, new_underlying_grid)
    return ImmersedBoundaryGrid(new_underlying_grid, new_immersed_boundary)
end

function scatter_local_grids(global_grid::ImmersedBoundaryGrid, arch::Distributed, local_size)
    ib = global_grid.immersed_boundary
    ug = global_grid.underlying_grid

    local_ug = scatter_local_grids(ug, arch, local_size)

    # Kinda hacky
    local_bottom_height = partition(ib.bottom_height, arch, local_size)
    ImmersedBoundaryConstructor = getnamewrapper(ib)
    local_ib = ImmersedBoundaryConstructor(local_bottom_height)
    
    return ImmersedBoundaryGrid(local_ug, local_ib)
end

"""
    function resize_immersed_boundary!(ib, grid)

If the immersed condition is an `OffsetArray`, resize it to match 
the total size of `grid`
"""
resize_immersed_boundary(ib::AbstractGridFittedBottom, grid) = ib
resize_immersed_boundary(ib::GridFittedBoundary, grid)       = ib

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

# A distributed grid with split interior map
const DistributedActiveCellsIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:DistributedGrid, <:Any, <:NamedTuple} 

# In case of a `DistributedGrid` we want to have different maps depending on the partitioning of the domain:
#
# If we partition the domain in the x-direction, we typically want to have the option to split three-dimensional 
# kernels in a `halo-independent` part in the range Hx+1:Nx-Hx, 1:Ny, 1:Nz and two `halo-dependent` computations:
# a west one spanning 1:Hx, 1:Ny, 1:Nz and an east one spanning Nx-Hx+1:Nx, 1:Ny, 1:Nz. 
# For this reason we need three different maps, one containing the `halo_independent` active region, a `west` map and an `east` map. 
# For the same reason we need to construct `south` and `north` maps if we partition the domain in the y-direction.
# Therefore, the `interior_active_cells` in this case is a `NamedTuple` containing 5 elements.
# Note that boundary-adjacent maps corresponding to non-partitioned directions are set to `nothing`
function map_interior_active_cells(ibg::ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:DistributedGrid})

    arch = architecture(ibg)

    # If we using a synchronized architecture, nothing
    # changes with serial execution.
    if arch isa SynchronizedDistributed
        return interior_active_indices(ibg; parameters = :xyz)
    end

    Rx, Ry, _  = arch.ranks
    Tx, Ty, _  = topology(ibg)
    Nx, Ny, Nz = size(ibg)
    Hx, Hy, _  = halo_size(ibg)
    
    x_boundary = (Hx, Ny, Nz)
    y_boundary = (Nx, Hy, Nz)
         
    left_offsets    = (0,  0,  0)
    right_x_offsets = (Nx-Hx, 0,     0)
    right_y_offsets = (0,     Ny-Hy, 0)

    include_west  = !isa(ibg, XFlatGrid) && (Rx != 1) && !(Tx == RightConnected)
    include_east  = !isa(ibg, XFlatGrid) && (Rx != 1) && !(Tx == LeftConnected)
    include_south = !isa(ibg, YFlatGrid) && (Ry != 1) && !(Ty == RightConnected)
    include_north = !isa(ibg, YFlatGrid) && (Ry != 1) && !(Ty == LeftConnected)

    west_halo_dependent_cells  = include_west  ? interior_active_indices(ibg; parameters = KernelParameters(x_boundary, left_offsets))    : nothing
    east_halo_dependent_cells  = include_east  ? interior_active_indices(ibg; parameters = KernelParameters(x_boundary, right_x_offsets)) : nothing
    south_halo_dependent_cells = include_south ? interior_active_indices(ibg; parameters = KernelParameters(y_boundary, left_offsets))    : nothing
    north_halo_dependent_cells = include_north ? interior_active_indices(ibg; parameters = KernelParameters(y_boundary, right_y_offsets)) : nothing
    
    nx = Rx == 1 ? Nx : (Tx == RightConnected || Tx == LeftConnected ? Nx - Hx : Nx - 2Hx)
    ny = Ry == 1 ? Ny : (Ty == RightConnected || Ty == LeftConnected ? Ny - Hy : Ny - 2Hy)

    ox = Rx == 1 || Tx == RightConnected ? 0 : Hx
    oy = Ry == 1 || Ty == RightConnected ? 0 : Hy
     
    halo_independent_cells = interior_active_indices(ibg; parameters = KernelParameters((nx, ny, Nz), (ox, oy, 0)))

    return (; halo_independent_cells, 
              west_halo_dependent_cells, 
              east_halo_dependent_cells, 
              south_halo_dependent_cells, 
              north_halo_dependent_cells)
end
