using Oceananigans.Utils: getnamewrapper
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: DistributedGrid, construct_global_array, global_size

import Oceananigans.DistributedComputations: reconstruct_global_grid, scatter_local_grids

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

function scatter_local_grids(arch::Distributed, global_grid::ImmersedBoundaryGrid, local_size)
    ib = global_grid.immersed_boundary
    ug = global_grid.underlying_grid

    local_ug = scatter_local_grids(arch, ug, local_size)
    local_ib = getnamewrapper(ib)(partition_global_array(arch, ib.bottom_height, local_size))
    
    return ImmersedBoundaryGrid(local_ug, local_ib)
end

"""
    resize_immersed_boundary!(ib, grid)

If the immersed condition is an `OffsetArray`, resize it to match 
the total size of `grid`.
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
        cpu_bottom   = arch_array(CPU(), ib.bottom_height)[1:Nx, 1:Ny] 
        set!(bottom_field, cpu_bottom)
        fill_halo_regions!(bottom_field)
        offset_bottom_array = dropdims(bottom_field.data, dims=3)

        return getnamewrapper(ib)(offset_bottom_array)
    end

    return ib
end
