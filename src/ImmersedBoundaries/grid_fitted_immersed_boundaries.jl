using Adapt
using CUDA: CuArray
using Oceananigans.Fields: ReducedField, fill_halo_regions!

abstract type AbstractGridFittedBoundary <: AbstractImmersedBoundary end

#####
##### GridFittedBoundary
#####

struct GridFittedBoundary{S} <: AbstractGridFittedBoundary
    mask :: S
end

@inline is_immersed(i, j, k, underlying_grid, ib::GridFittedBoundary) = ib.mask(node(c, c, c, i, j, k, underlying_grid)...)

#####
##### GridFittedBottom
#####

"""
    GridFittedBottom(bottom)

Return an immersed boundary...
"""
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

const GFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBottom}

@inline Δzᵃᵃᶜ(i, j, k, ibg::GFBIBG) = ifelse(is_immersed(i, j, k, ibg.grid, ibg.immersed_boundary),
                                             zero(eltype(ibg.grid)),
                                             Δzᵃᵃᶜ(i, j, k, ibg.grid))

@inline Δzᵃᵃᶠ(i, j, k, ibg::GFBIBG) = ifelse(is_immersed(i, j, k, ibg.grid, ibg.immersed_boundary),
                                             zero(eltype(ibg.grid)),
                                             Δzᵃᵃᶠ(i, j, k, ibg.grid))

Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.bottom))     

