using Adapt
using CUDA: CuArray
using OffsetArrays: OffsetArray
using Oceananigans.Utils: getnamewrapper
using Oceananigans.Grids: total_size, rnode
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

struct GridFittedBottom{H, I} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    immersed_condition :: I
end

Base.summary(::CenterImmersedCondition) = "CenterImmersedCondition"
Base.summary(::InterfaceImmersedCondition) = "InterfaceImmersedCondition"

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
end

on_architecture(arch, ib::GridFittedBottom) = GridFittedBottom(on_architecture(arch, ib.bottom_height), ib.immersed_condition)

function on_architecture(arch, ib::GridFittedBottom{<:Field})
    architecture(ib.bottom_height) == arch && return ib
    arch_grid = on_architecture(arch, ib.bottom_height.grid)
    new_bottom_height = Field{Center, Center, Nothing}(arch_grid)
    set!(new_bottom_height, ib.bottom_height)
    fill_halo_regions!(new_bottom_height)
    return GridFittedBottom(new_bottom_height, ib.immersed_condition)
end

Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.bottom_height), adapt(to, ib.immersed_condition))

"""
    materialize_immersed_boundary(grid, ib)

Returns a new `ib` wrapped around a Field that holds the numerical `immersed_boundary`.
If `ib` is an `AbstractGridFittedBottom`, `ib.bottom_height` is the z-coordinate of
top-most interface of the last ``immersed`` cell in the column. If `ib` is a `GridFittedBoundary`,
`ib.mask` is a field of booleans that indicates whether a cell is immersed or not.
"""
function materialize_immersed_boundary(grid, ib::GridFittedBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.bottom_height)
    @apply_regionally compute_numerical_bottom_height!(bottom_field, grid, ib)
    fill_halo_regions!(bottom_field)
    new_ib = GridFittedBottom(bottom_field)
    return new_ib
end

compute_numerical_bottom_height!(bottom_field, grid, ib) =
    launch!(architecture(grid), grid, :xy, _compute_numerical_bottom_height!, bottom_field, grid, ib)

@kernel function _compute_numerical_bottom_height!(bottom_field, grid, ib::GridFittedBottom)
    i, j = @index(Global, NTuple)
    zb = @inbounds bottom_field[i, j, 1]
    @inbounds bottom_field[i, j, 1] = rnode(i, j, 1, grid, c, c, f)
    condition = ib.immersed_condition
    for k in 1:grid.Nz
        z⁺ = rnode(i, j, k+1, grid, c, c, f)
        z  = rnode(i, j, k,   grid, c, c, c)
        bottom_cell = ifelse(condition isa CenterImmersedCondition, z ≤ zb, z⁺ ≤ zb)
        @inbounds bottom_field[i, j, 1] = ifelse(bottom_cell, z⁺, bottom_field[i, j, 1])
    end
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom)
    # We use `rnode` for the `immersed_cell` because we do not want to have
    # wetting or drying that could happen for a moving grid if we use znode
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zb = @inbounds ib.bottom_height[i, j, 1]
    return z ≤ zb
end

@inline function _immersed_cell(i, j, k::AbstractArray, underlying_grid, ib::GridFittedBottom)
    # We use `rnode` for the `immersed_cell` because we do not want to have
    # wetting or drying that could happen for a moving grid if we use znode
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zb = @inbounds ib.bottom_height[i, j, 1]
    zb = Base.stack(collect(zb for _ in k))
    return z .≤ zb
end

#####
##### Static column depth
#####

const AGFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}

@inline static_column_depthᶜᶜᵃ(i, j, ibg::AGFBIBG) = @inbounds rnode(i, j, ibg.Nz+1, ibg, c, c, f) - ibg.immersed_boundary.bottom_height[i, j, 1]
@inline static_column_depthᶜᶠᵃ(i, j, ibg::AGFBIBG) = min(static_column_depthᶜᶜᵃ(i, j-1, ibg), static_column_depthᶜᶜᵃ(i, j, ibg))
@inline static_column_depthᶠᶜᵃ(i, j, ibg::AGFBIBG) = min(static_column_depthᶜᶜᵃ(i-1, j, ibg), static_column_depthᶜᶜᵃ(i, j, ibg))
@inline static_column_depthᶠᶠᵃ(i, j, ibg::AGFBIBG) = min(static_column_depthᶠᶜᵃ(i, j-1, ibg), static_column_depthᶠᶜᵃ(i, j, ibg))

# Make sure column_height works for horizontally-Flat topologies.
XFlatAGFIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}
YFlatAGFIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:AbstractGridFittedBottom}

@inline static_column_depthᶠᶜᵃ(i, j, ibg::XFlatAGFIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::YFlatAGFIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)
@inline static_column_depthᶠᶠᵃ(i, j, ibg::XFlatAGFIBG) = static_column_depthᶜᶠᵃ(i, j, ibg)
@inline static_column_depthᶠᶠᵃ(i, j, ibg::YFlatAGFIBG) = static_column_depthᶠᶜᵃ(i, j, ibg)
