using CUDA: CuArray
using Oceananigans.Fields: ReducedField, fill_halo_regions!

abstract type AbstractGridFittedBoundary <: AbstractImmersedBoundary end

#####
##### GridFittedBoundary
#####

struct RasterDepthMask end

struct GridFittedBoundary{M, S} <: AbstractGridFittedBoundary
    mask :: S
    mask_type :: M
end

GridFittedBoundary(mask; mask_type=nothing) = GridFittedBoundary(mask, mask_type)

@inline is_immersed(i, j, k, underlying_grid, ib::GridFittedBoundary) = ib.mask(node(c, c, c, i, j, k, underlying_grid)...)
@inline is_immersed(i, j, k, underlying_grid, ib::GridFittedBoundary{<:RasterDepthMask}) = ib.mask(i, j)

#####
##### GridFittedBottom
#####

struct GridFittedBottom{B} <: AbstractGridFittedBoundary
    bottom :: B
end

@inline function is_immersed(i, j, k, underlying_grid, ib::GridFittedBottom)
    x, y, z = node(c, c, c, i, j, k, underlying_grid)
    return z < ib.bottom(x, y)
end

@inline function is_immersed(i, j, k, underlying_grid, ib::GridFittedBottom{<:AbstractArray})
    x, y, z = node(c, c, c, i, j, k, underlying_grid)
    return z < ib.bottom[i, j]
end

const ArrayGridFittedBottom = GridFittedBottom{<:Array}
const CuArrayGridFittedBottom = GridFittedBottom{<:CuArray}

function ImmersedBoundaryGrid(grid, ib::Union{ArrayGridFittedBottom, CuArrayGridFittedBottom})
    # Wrap bathymetry in an OffsetArray with halos
    arch = architecture(ib.bottom)
    bottom_field = ReducedField(Center, Center, Nothing, arch, grid; dims=3)
    bottom_field .= ib.bottom
    fill_halo_regions!(bottom_field, arch)
    offset_bottom_array = dropdims(bottom_field.data, dims=3)
    new_ib = GridFittedBottom(offset_bottom_array)
    return ImmersedBoundaryGrid(grid, new_ib)
end

