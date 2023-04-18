using Adapt
using CUDA: CuArray
using OffsetArrays: OffsetArray
using Oceananigans.Utils: getnamewrapper
using Oceananigans.Grids: total_size
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.Architectures: arch_array
using Oceananigans.BoundaryConditions: FBC
using Printf

import Oceananigans.TurbulenceClosures: z_bottom

#####
##### GridFittedBottom (2.5D immersed boundary with modified bottom height)
#####

abstract type AbstractGridFittedBottom{H} <: AbstractGridFittedBoundary end

# To enable comparison with PartialCellBottom in the limiting case that
# fractional cell height is 1.0.
struct CenterImmersedCondition end
struct InterfaceImmersedCondition end

Base.summary(::CenterImmersedCondition) = "CenterImmersedCondition"
Base.summary(::InterfaceImmersedCondition) = "InterfaceImmersedCondition"

"""
    GridFittedBottom(bottom_height, [immersed_condition=CenterImmersedCondition()])

Return an immersed boundary with an irregular bottom fit to the underlying grid.
"""
struct GridFittedBottom{H, I} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    immersed_condition :: I
end

const GFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBottom}

GridFittedBottom(bottom_height) = GridFittedBottom(bottom_height, CenterImmersedCondition())

function Base.summary(ib::GridFittedBottom)
    hmax = maximum(interior(ib.bottom_height))
    hmin = minimum(interior(ib.bottom_height))
    hmean = mean(interior(ib.bottom_height))

    summary1 = "GridFittedBottom("

    summary2 = string("mean(z)=", prettysummary(hmean),
                      ", min(z)=", prettysummary(hmin),
                      ", max(z)=", prettysummary(hmax))

    summary3 = ")"

    return summary1 * summary2 * summary3
end

Base.summary(ib::GridFittedBottom{<:Function}) = @sprintf("GridFittedBottom(%s)", ib.bottom_height)

function Base.show(io::IO, ib::GridFittedBottom)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "└── immersed_condition: ", summary(ib.immersed_condition))
end

"""
    ImmersedBoundaryGrid(grid, ib::GridFittedBottom)

Return a grid with `GridFittedBottom` immersed boundary.

Computes ib.bottom_height and wraps in an array.
"""
function ImmersedBoundaryGrid(grid, ib::GridFittedBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.bottom_height)
    fill_halo_regions!(bottom_field)
    new_ib = GridFittedBottom(bottom_field, ib.immersed_condition)
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, new_ib)
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom{<:Any, <:InterfaceImmersedCondition})
    z = znode(i, j, k+1, underlying_grid, c, c, f)
    h = @inbounds ib.bottom_height[i, j, 1]
    return z <= h
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom{<:Any, <:CenterImmersedCondition})
    z = znode(i, j, k, underlying_grid, c, c, c)
    h = @inbounds ib.bottom_height[i, j, 1]
    return z <= h
end

@inline z_bottom(i, j, ibg::GFBIBG) = @inbounds ibg.immersed_boundary.bottom_height[i, j, 1]
on_architecture(arch, ib::GridFittedBottom) = GridFittedBottom(ib.bottom_height, ib.immersed_condition)

function on_architecture(arch, ib::GridFittedBottom{<:Field})
    architecture(ib.bottom_height) == arch && return ib
    arch_grid = on_architecture(arch, ib.bottom_height.grid)
    new_bottom_height = Field{Center, Center, Nothing}(arch_grid)
    copyto!(parent(new_bottom_height), parent(ib.bottom_height))
    return GridFittedBottom(new_bottom_height, ib.immersed_condition)
end

Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.bottom_height.data),
                                                                   ib.immersed_condition)

