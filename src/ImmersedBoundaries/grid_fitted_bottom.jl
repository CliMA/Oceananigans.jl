using Oceananigans.Grids: Grids, constructor_arguments, rnode
using Oceananigans.Fields: Field, fill_halo_regions!, interior
using Oceananigans.BoundaryConditions: FBC
using OffsetArrays: OffsetArray

#####
##### GridFittedBottom (2.5D immersed boundary with modified bottom height)
#####

abstract type AbstractGridFittedBottom{H} <: AbstractGridFittedBoundary end

# To enable comparison with PartialCellBottom in the limiting case that
# fractional cell height is 1.0.
struct CenterImmersedCondition end
struct InterfaceImmersedCondition end

struct GridFittedBottom{H, T, I} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    top_height :: T
    immersed_condition :: I
end

GridFittedBottom(bottom_height, immersed_condition) = GridFittedBottom(bottom_height, nothing, immersed_condition)

Base.summary(::CenterImmersedCondition) = "CenterImmersedCondition"
Base.summary(::InterfaceImmersedCondition) = "InterfaceImmersedCondition"

const GFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBottom}

"""
    GridFittedBottom(bottom_height=nothing; top_height=nothing, immersed_condition=CenterImmersedCondition())

Return a bottom immersed boundary, optionally with an immersed top.

Arguments
=========

* `bottom_height`: an array or function that gives the height of the
                   bottom in absolute ``z`` coordinates. Default: `nothing`
                   (the bottom of the domain, e.g. for a flat-bottomed domain with only a top).

Keyword arguments
=================

* `top_height`: an array or function that gives the height of the underside of
                a solid top (e.g. an ice-shelf draft) in absolute ``z`` coordinates.
                Cells above `top_height` are immersed. Default: `nothing` (no top).

* `immersed_condition`: Determine whether the part of the domain that is
                        immersed are all the cell centers that lie below
                        `bottom_height` (`CenterImmersedCondition()`; default)
                        or all the cell faces that lie below `bottom_height`
                        (`InterfaceImmersedCondition()`). The only purpose of
                        `immersed_condition` to allow `GridFittedBottom` and
                        `PartialCellBottom` to have the same behavior when the
                        minimum fractional cell height for partial cells is set
                        to 0. The same condition applies to the top.

Columns in which the top lies at or below the bottom are immersed entirely.

Example
=======

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 8, 10), x=(0, 100), y=(0, 100), z=(-100, 0));

julia> ImmersedBoundaryGrid(grid, GridFittedBottom(-90; top_height=(x, y) -> -20 - 0.2y))
2×8×10 ImmersedBoundaryGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo:
├── immersed_boundary: GridFittedBottom(mean(z)=-90.0, min(z)=-90.0, max(z)=-90.0, mean(zt)=-30.0, min(zt)=-40.0, max(zt)=-20.0)
├── underlying_grid: 2×8×10 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── Periodic x ∈ [0.0, 100.0)  regularly spaced with Δx=50.0
├── Periodic y ∈ [0.0, 100.0)  regularly spaced with Δy=12.5
└── Bounded  z ∈ [-100.0, 0.0] regularly spaced with Δz=10.0
```
"""
GridFittedBottom(bottom_height=nothing; top_height=nothing, immersed_condition=CenterImmersedCondition()) =
    GridFittedBottom(bottom_height, top_height, immersed_condition)

# 1-based interior view of a bare bottom-height array.
@inline function bottom_height_interior(bottom_height)
    parent_ranges = ntuple(Val(ndims(bottom_height))) do d
        H = 1 - first(axes(bottom_height, d))
        (1 + H):(size(bottom_height, d) - H)
    end
    return view(parent(bottom_height), parent_ranges...)
end

@inline bottom_heights_equal(h1, h2) = h1 == h2
@inline bottom_heights_equal(h1::AbstractArray, h2::AbstractArray) = bottom_height_interior(h1) == bottom_height_interior(h2)

set_bottom_height!(bottom_field, bottom_height) = set!(bottom_field, bottom_height)

# Without a bottom, the numerical bottom height becomes the bottom of the domain
set_bottom_height!(bottom_field, ::Nothing) = set!(bottom_field, -Inf)

function set_bottom_height!(bottom_field, bottom_height::OffsetArray)
    source = on_architecture(architecture(bottom_field), bottom_height_interior(bottom_height))
    copyto!(interior(bottom_field), source)
    return bottom_field
end

bottom_height_field(bottom_data, grid) = Field{Center, Center, Nothing}(grid; data=bottom_data)

"""
$(TYPEDSIGNATURES)

Return a `Field` at `(Center, Center, Nothing)` that wraps the bottom height of `grid`, which is
stored on `grid.immersed_boundary` as a bare `OffsetArray`. The returned `Field` shares its data
with `grid`, so mutating it mutates the bottom height of `grid`.
"""
bottom_height_field(grid::IBG) = bottom_height_field(grid.immersed_boundary.bottom_height, grid.underlying_grid)

"""
$(TYPEDSIGNATURES)

Return a `Field` at `(Center, Center, Nothing)` that wraps the top height of `grid`, or `nothing`
if `grid` has no immersed top. The returned `Field` shares its data with `grid`.
"""
top_height_field(grid::IBG) = top_height_field(grid.immersed_boundary.top_height, grid.underlying_grid)
top_height_field(::Nothing, grid) = nothing
top_height_field(top_data, grid) = bottom_height_field(top_data, grid)

top_height_summary(::Nothing) = ""

function top_height_summary(top_height)
    top_interior = bottom_height_interior(top_height)
    zmean = sum(top_interior) / length(top_interior)
    return string(", mean(zt)=", prettysummary(zmean),
                  ", min(zt)=", prettysummary(minimum(top_interior)),
                  ", max(zt)=", prettysummary(maximum(top_interior)))
end

top_height_summary(top_height::Function) = string(", top_height=", prettysummary(top_height, false))

function Base.summary(ib::GridFittedBottom)
    bottom_interior = bottom_height_interior(ib.bottom_height)
    zmax  = maximum(bottom_interior)
    zmin  = minimum(bottom_interior)
    zmean = sum(bottom_interior) / length(bottom_interior)

    summary1 = "GridFittedBottom("

    summary2 = string("mean(z)=", prettysummary(zmean),
                      ", min(z)=", prettysummary(zmin),
                      ", max(z)=", prettysummary(zmax),
                      top_height_summary(ib.top_height))

    summary3 = ")"

    return summary1 * summary2 * summary3
end

Base.summary(ib::GridFittedBottom{<:Union{Function, Nothing}}) = @sprintf("GridFittedBottom(%s%s)", ib.bottom_height, top_height_summary(ib.top_height))

function Base.show(io::IO, ib::GridFittedBottom{<:Any, Nothing})
    print(io, summary(ib), '\n')
    print(io, "└── bottom_height: ", prettysummary(ib.bottom_height), '\n')
end

function Base.show(io::IO, ib::GridFittedBottom)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "└── top_height: ", prettysummary(ib.top_height), '\n')
end

Architectures.on_architecture(arch, ib::GridFittedBottom) = GridFittedBottom(on_architecture(arch, ib.bottom_height),
                                                                             on_architecture(arch, ib.top_height),
                                                                             ib.immersed_condition)

Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.bottom_height),
                                                                   adapt(to, ib.top_height),
                                                                   adapt(to, ib.immersed_condition))

"""
$(TYPEDSIGNATURES)

Returns a new `ib` that holds the numerical `immersed_boundary`.
If `ib` is an `AbstractGridFittedBottom`, `ib.bottom_height` is an `OffsetArray` holding the
z-coordinate of the top-most interface of the last ``immersed`` cell in the column (wrap it as a
`Field` with [`bottom_height_field`](@ref)). If `ib` is a `GridFittedBoundary`, `ib.mask` is a `Field` of
booleans that indicates whether a cell is immersed or not.
"""
function materialize_immersed_boundary(grid, ib::GridFittedBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(bottom_field, ib.bottom_height)
    top_field = materialize_top_height(grid, ib.top_height)

    compute_ib = GridFittedBottom(bottom_field, top_field, ib.immersed_condition)

    @apply_regionally compute_numerical_bottom_height!(bottom_field, grid, compute_ib)
    @apply_regionally compute_numerical_top_height!(bottom_field, top_field, grid, compute_ib)
    fill_halo_regions!(bottom_field)
    fill_top_height_halo_regions!(top_field)

    return GridFittedBottom(bottom_field.data, top_height_data(top_field), ib.immersed_condition)
end

materialize_top_height(grid, ::Nothing) = nothing

function materialize_top_height(grid, top_height)
    top_field = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(top_field, top_height)
    return top_field
end

fill_top_height_halo_regions!(::Nothing) = nothing
fill_top_height_halo_regions!(top_field) = fill_halo_regions!(top_field)

top_height_data(::Nothing) = nothing
top_height_data(top_field) = top_field.data

compute_numerical_bottom_height!(bottom_field, grid, ib) =
    launch!(architecture(grid), grid, :xy, _compute_numerical_bottom_height!, bottom_field, grid, ib)

compute_numerical_top_height!(bottom_field, ::Nothing, grid, ib) = nothing

compute_numerical_top_height!(bottom_field, top_field, grid, ib) =
    launch!(architecture(grid), grid, :xy, _compute_numerical_top_height!, bottom_field, top_field, grid, ib)

@kernel function _compute_numerical_bottom_height!(bottom_field, grid, ib::GridFittedBottom)
    i, j = @index(Global, NTuple)
    zb = @inbounds bottom_field[i, j, 1]
    @inbounds bottom_field[i, j, 1] = rnode(i, j, 1, grid, c, c, f)
    condition = ib.immersed_condition
    for k in 1:grid.Nz
        z⁺ = rnode(i, j, k+1, grid, c, c, f)
        z  = rnode(i, j, k,   grid, c, c, c)
        immersed_cell = ifelse(condition isa CenterImmersedCondition, z ≤ zb, z⁺ ≤ zb)
        @inbounds bottom_field[i, j, 1] = ifelse(immersed_cell, z⁺, bottom_field[i, j, 1])
    end
end

@kernel function _compute_numerical_top_height!(bottom_field, top_field, grid, ib::GridFittedBottom)
    i, j = @index(Global, NTuple)
    zᵗ = @inbounds top_field[i, j, 1]
    ẑᵗ = rnode(i, j, grid.Nz+1, grid, c, c, f)
    condition = ib.immersed_condition
    for k in 1:grid.Nz
        z⁻ = rnode(i, j, k, grid, c, c, f)
        z  = rnode(i, j, k, grid, c, c, c)
        immersed_cell = ifelse(condition isa CenterImmersedCondition, z ≥ zᵗ, z⁻ ≥ zᵗ)
        ẑᵗ = ifelse(immersed_cell, min(ẑᵗ, z⁻), ẑᵗ)
    end

    # Close columns in which the top lies at or below the bottom
    domain_top = rnode(i, j, grid.Nz+1, grid, c, c, f)
    ẑᵇ = @inbounds bottom_field[i, j, 1]
    closed = ẑᵗ ≤ ẑᵇ
    @inbounds bottom_field[i, j, 1] = ifelse(closed, domain_top, ẑᵇ)
    @inbounds top_field[i, j, 1] = ifelse(closed, domain_top, ẑᵗ)
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom)
    # We use `rnode` for the `immersed_cell` because we do not want to have
    # wetting or drying that could happen for a moving grid if we use znode
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zb = @inbounds ib.bottom_height[i, j, 1]
    return (z ≤ zb) | above_top(i, j, z, ib.top_height)
end

@inline above_top(i, j, z, ::Nothing) = false
@inline above_top(i, j, z, top_height) = @inbounds z ≥ top_height[i, j, 1]

@inline function _immersed_cell(i, j, k::AbstractArray, underlying_grid, ib::GridFittedBottom{<:Any, Nothing})
    # We use `rnode` for the `immersed_cell` because we do not want to have
    # wetting or drying that could happen for a moving grid if we use znode
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zb = @inbounds ib.bottom_height[i, j, 1]
    _zb = Base.stack(collect(zb for _ in k))
    return z .≤ _zb
end

@inline function _immersed_cell(i, j, k::AbstractArray, underlying_grid, ib::GridFittedBottom)
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zb = @inbounds ib.bottom_height[i, j, 1]
    zt = @inbounds ib.top_height[i, j, 1]
    return (z .≤ zb) .| (z .≥ zt)
end

#####
##### Static column depth
#####

# AbstractGridFittedBottomImmersedBoundaryGrid
const AGFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}

@inline static_column_depthᶜᶜᵃ(i, j, ibg::AGFBIBG) = @inbounds column_top_height(i, j, ibg, ibg.immersed_boundary.top_height) - ibg.immersed_boundary.bottom_height[i, j, 1]

@inline column_top_height(i, j, grid, ::Nothing) = rnode(i, j, grid.Nz+1, grid, c, c, f)
@inline column_top_height(i, j, grid, top_height) = @inbounds top_height[i, j, 1]
@inline static_column_depthᶜᶠᵃ(i, j, ibg::AGFBIBG) = min(static_column_depthᶜᶜᵃ(i, j-1, ibg), static_column_depthᶜᶜᵃ(i, j, ibg))
@inline static_column_depthᶠᶜᵃ(i, j, ibg::AGFBIBG) = min(static_column_depthᶜᶜᵃ(i-1, j, ibg), static_column_depthᶜᶜᵃ(i, j, ibg))
@inline static_column_depthᶠᶠᵃ(i, j, ibg::AGFBIBG) = min(static_column_depthᶠᶜᵃ(i, j-1, ibg), static_column_depthᶠᶜᵃ(i, j, ibg))

# Make sure column_height works for horizontally-Flat topologies.
const XFlatAGFIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}
const YFlatAGFIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:AbstractGridFittedBottom}
const XYFlatAGFIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Flat, <:Any, <:Any, <:AbstractGridFittedBottom}

@inline static_column_depthᶠᶜᵃ(i, j, ibg::XFlatAGFIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::YFlatAGFIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)
@inline static_column_depthᶠᶠᵃ(i, j, ibg::XFlatAGFIBG) = static_column_depthᶜᶠᵃ(i, j, ibg)
@inline static_column_depthᶠᶠᵃ(i, j, ibg::YFlatAGFIBG) = static_column_depthᶠᶜᵃ(i, j, ibg)
@inline static_column_depthᶠᶠᵃ(i, j, ibg::XYFlatAGFIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)

function Grids.constructor_arguments(grid::AGFBIBG)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid.underlying_grid)
    grid_fitted_bottom_args = Dict(:bottom_height      => grid.immersed_boundary.bottom_height,
                                   :immersed_condition => grid.immersed_boundary.immersed_condition)
    return underlying_grid_args, underlying_grid_kwargs, grid_fitted_bottom_args
end

function Base.:(==)(gfb1::GridFittedBottom, gfb2::GridFittedBottom)
    return bottom_heights_equal(gfb1.bottom_height, gfb2.bottom_height) &&
           bottom_heights_equal(gfb1.top_height, gfb2.top_height) &&
           gfb1.immersed_condition == gfb2.immersed_condition
end
