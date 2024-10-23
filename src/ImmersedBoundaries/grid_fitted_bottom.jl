using Adapt
using CUDA: CuArray
using OffsetArrays: OffsetArray
using Oceananigans.Utils: getnamewrapper
using Oceananigans.Grids: total_size
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.BoundaryConditions: FBC
using Printf

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

struct GridFittedBottom{H, I} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    immersed_condition :: I
end

const GFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBottom}

"""
    GridFittedBottom(bottom_height, [immersed_condition=CenterImmersedCondition()])

Return a bottom immersed boundary.

Keyword Arguments
=================

* `bottom_height`: an array or function that gives the height of the
                   bottom in absolute ``z`` coordinates.

* `immersed_condition`: Determine whether the part of the domain that is 
                        immersed are all the cell centers that lie below
                        `bottom_height` (`CenterImmersedCondition()`; default)
                        or all the cell faces that lie below `bottom_height`
                        (`InterfaceImmersedCondition()`). The only purpose of
                        `immersed_condition` to allow `GridFittedBottom` and
                        `PartialCellBottom` to have the same behavior when the
                        minimum fractional cell height for partial cells is set
                        to 0.
"""
GridFittedBottom(bottom_height) = GridFittedBottom(bottom_height, CenterImmersedCondition())

function Base.summary(ib::GridFittedBottom)
    zmax  = maximum(ib.bottom_height)
    zmin  = minimum(ib.bottom_height)
    zmean = mean(ib.bottom_height)

    summary1 = "GridFittedBottom("

    summary2 = string("mean(z)=", prettysummary(zmean),
                      ", min(z)=", prettysummary(zmin),
                      ", max(z)=", prettysummary(zmax))

    summary3 = ")"

    return summary1 * summary2 * summary3
end

Base.summary(ib::GridFittedBottom{<:Function}) = @sprintf("GridFittedBottom(%s)", ib.bottom_height)

function Base.show(io::IO, ib::GridFittedBottom)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "└── immersed_condition: ", summary(ib.immersed_condition))
end

@inline z_bottom(i, j, ibg::GFBIBG) = @inbounds ibg.immersed_boundary.bottom_height[i, j, 1]

"""
    ImmersedBoundaryGrid(grid, ib::GridFittedBottom)

Return a grid with `GridFittedBottom` immersed boundary (`ib`).

Computes `ib.bottom_height` and wraps it in a Field.
"""
function ImmersedBoundaryGrid(grid, ib::GridFittedBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.bottom_height)
    @apply_regionally clamp_bottom_height!(bottom_field, grid)
    fill_halo_regions!(bottom_field)
    new_ib = GridFittedBottom(bottom_field, ib.immersed_condition)
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, new_ib)
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom{<:Any, <:InterfaceImmersedCondition})
    z = znode(i, j, k+1, underlying_grid, c, c, f)
    h = @inbounds ib.bottom_height[i, j, 1]
    return z ≤ h
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom{<:Any, <:CenterImmersedCondition})
    z = znode(i, j, k, underlying_grid, c, c, c)
    h = @inbounds ib.bottom_height[i, j, 1]
    return z ≤ h
end

on_architecture(arch, ib::GridFittedBottom) = GridFittedBottom(ib.bottom_height, ib.immersed_condition)

function on_architecture(arch, ib::GridFittedBottom{<:Field})
    architecture(ib.bottom_height) == arch && return ib
    arch_grid = on_architecture(arch, ib.bottom_height.grid)
    new_bottom_height = Field{Center, Center, Nothing}(arch_grid)
    set!(new_bottom_height, ib.bottom_height)
    fill_halo_regions!(new_bottom_height)
    return GridFittedBottom(new_bottom_height, ib.immersed_condition)
end

Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.bottom_height),
                                                                             ib.immersed_condition)
