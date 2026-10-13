using Oceananigans.Grids: Grids, constructor_arguments, rnode
using Oceananigans.Fields: Field, fill_halo_regions!, interior
using Oceananigans.BoundaryConditions: FBC
using OffsetArrays: OffsetArray

#####
##### GridFittedVerticalBoundaries (2.5D immersed boundary with modified bottom and top heights)
#####

abstract type AbstractGridFittedBottom{H} <: AbstractGridFittedBoundary end

# To enable comparison with PartialCellBottom in the limiting case that
# fractional cell height is 1.0.
struct CenterImmersedCondition end
struct InterfaceImmersedCondition end

struct GridFittedVerticalBoundaries{H, T, I} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    top_height :: T
    immersed_condition :: I
end

# A missing top is materialized as a `ConstantField` at the top of the domain
const GridFittedBottom{H, I} = GridFittedVerticalBoundaries{H, <:Union{Nothing, ConstantField}, I}

GridFittedBottom(bottom_height, immersed_condition) = GridFittedVerticalBoundaries(bottom_height, nothing, immersed_condition)

Base.summary(::CenterImmersedCondition) = "CenterImmersedCondition"
Base.summary(::InterfaceImmersedCondition) = "InterfaceImmersedCondition"

const GFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedVerticalBoundaries}

"""
    GridFittedVerticalBoundaries(; bottom_height=nothing, top_height=nothing, immersed_condition=CenterImmersedCondition())

Return an immersed boundary that bounds each column from below by `bottom_height`
and from above by `top_height`.

Keyword arguments
=================

* `bottom_height`: an array or function that gives the height of the
                   bottom in absolute ``z`` coordinates. Default: `nothing`
                   (the bottom of the domain, e.g. for a flat-bottomed domain with only a top).

* `top_height`: an array or function that gives the height of the underside of
                a solid top (e.g. an ice-shelf draft) in absolute ``z`` coordinates.
                Cells above `top_height` are immersed. Default: `nothing` (no top).

* `immersed_condition`: Determine whether the part of the domain that is
                        immersed are all the cell centers that lie below
                        `bottom_height` (`CenterImmersedCondition()`; default)
                        or all the cell faces that lie below `bottom_height`
                        (`InterfaceImmersedCondition()`). The only purpose of
                        `immersed_condition` to allow `GridFittedVerticalBoundaries` and
                        `PartialCellVerticalBoundaries` to have the same behavior when the
                        minimum fractional cell height for partial cells is set
                        to 0. The same condition applies to the top.

Columns in which the top lies at or below the bottom are immersed entirely.

Example
=======

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 8, 10), x=(0, 100), y=(0, 100), z=(-100, 0));

julia> ImmersedBoundaryGrid(grid, GridFittedVerticalBoundaries(bottom_height=-90, top_height=(x, y) -> -20 - 0.2y))
2×8×10 ImmersedBoundaryGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo:
├── immersed_boundary: GridFittedVerticalBoundaries(mean(z)=-90.0, min(z)=-90.0, max(z)=-90.0, mean(zt)=-30.0, min(zt)=-40.0, max(zt)=-20.0)
├── underlying_grid: 2×8×10 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── Periodic x ∈ [0.0, 100.0)  regularly spaced with Δx=50.0
├── Periodic y ∈ [0.0, 100.0)  regularly spaced with Δy=12.5
└── Bounded  z ∈ [-100.0, 0.0] regularly spaced with Δz=10.0
```
"""
GridFittedVerticalBoundaries(; bottom_height=nothing, top_height=nothing, immersed_condition=CenterImmersedCondition()) =
    GridFittedVerticalBoundaries(bottom_height, top_height, immersed_condition)

"""
    GridFittedBottom(bottom_height=nothing; immersed_condition=CenterImmersedCondition())

Return a [`GridFittedVerticalBoundaries`](@ref) without an immersed top, bounded below by `bottom_height`.

Example
=======

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 8, 10), x=(0, 100), y=(0, 100), z=(-100, 0));

julia> ImmersedBoundaryGrid(grid, GridFittedBottom(-90))
2×8×10 ImmersedBoundaryGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo:
├── immersed_boundary: GridFittedBottom(mean(z)=-90.0, min(z)=-90.0, max(z)=-90.0)
├── underlying_grid: 2×8×10 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── Periodic x ∈ [0.0, 100.0)  regularly spaced with Δx=50.0
├── Periodic y ∈ [0.0, 100.0)  regularly spaced with Δy=12.5
└── Bounded  z ∈ [-100.0, 0.0] regularly spaced with Δz=10.0
```
"""
GridFittedBottom(bottom_height=nothing; immersed_condition=CenterImmersedCondition()) =
    GridFittedVerticalBoundaries(bottom_height, nothing, immersed_condition)

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
@inline bottom_heights_equal(h1::ConstantField, h2::ConstantField) = h1.constant == h2.constant
@inline bottom_heights_equal(h1::ConstantField, h2::AbstractArray) = false
@inline bottom_heights_equal(h1::AbstractArray, h2::ConstantField) = false

set_bottom_height!(bottom_field, bottom_height) = set!(bottom_field, bottom_height)

function set_bottom_height!(bottom_field, bottom_height::OffsetArray)
    source = on_architecture(architecture(bottom_field), bottom_height_interior(bottom_height))
    copyto!(interior(bottom_field), source)
    return bottom_field
end

bottom_height_field(bottom_data, grid) = Field{Center, Center, Nothing}(grid; data=bottom_data)

function bottom_height_field(height::ConstantField, grid)
    height_field = Field{Center, Center, Nothing}(grid)
    set!(height_field, height.constant)
    return height_field
end

"""
$(TYPEDSIGNATURES)

Return a `Field` at `(Center, Center, Nothing)` that wraps the bottom height of `grid`, which is
stored on `grid.immersed_boundary` as a bare `OffsetArray`. The returned `Field` shares its data
with `grid`, so mutating it mutates the bottom height of `grid`. If `grid` has no immersed bottom,
the returned `Field` is a copy filled with the bottom of the domain.
"""
bottom_height_field(grid::IBG) = bottom_height_field(grid.immersed_boundary.bottom_height, grid.underlying_grid)

"""
$(TYPEDSIGNATURES)

Return a `Field` at `(Center, Center, Nothing)` that wraps the top height of `grid`. The returned
`Field` shares its data with `grid`. If `grid` has no immersed top, the returned `Field` is a copy
filled with the top of the domain.
"""
top_height_field(grid::IBG) = bottom_height_field(grid.immersed_boundary.top_height, grid.underlying_grid)

function height_statistics(height)
    height_interior = bottom_height_interior(height)
    mean_height = sum(height_interior) / length(height_interior)
    return mean_height, minimum(height_interior), maximum(height_interior)
end

height_statistics(height::ConstantField) = (height.constant, height.constant, height.constant)

top_height_summary(::Union{Nothing, ConstantField}) = ""

function top_height_summary(top_height)
    zmean, zmin, zmax = height_statistics(top_height)
    return string(", mean(zt)=", prettysummary(zmean),
                  ", min(zt)=", prettysummary(zmin),
                  ", max(zt)=", prettysummary(zmax))
end

top_height_summary(top_height::Function) = string(", top_height=", prettysummary(top_height, false))

immersed_boundary_name(::GridFittedBottom) = "GridFittedBottom"
immersed_boundary_name(::GridFittedVerticalBoundaries) = "GridFittedVerticalBoundaries"

function Base.summary(ib::GridFittedVerticalBoundaries)
    zmean, zmin, zmax = height_statistics(ib.bottom_height)

    summary1 = immersed_boundary_name(ib) * "("

    summary2 = string("mean(z)=", prettysummary(zmean),
                      ", min(z)=", prettysummary(zmin),
                      ", max(z)=", prettysummary(zmax),
                      top_height_summary(ib.top_height))

    summary3 = ")"

    return summary1 * summary2 * summary3
end

Base.summary(ib::GridFittedVerticalBoundaries{<:Union{Function, Nothing}}) =
    @sprintf("%s(%s%s)", immersed_boundary_name(ib), ib.bottom_height, top_height_summary(ib.top_height))

function Base.show(io::IO, ib::GridFittedBottom)
    print(io, summary(ib), '\n')
    print(io, "└── bottom_height: ", prettysummary(ib.bottom_height), '\n')
end

function Base.show(io::IO, ib::GridFittedVerticalBoundaries)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "└── top_height: ", prettysummary(ib.top_height), '\n')
end

Architectures.on_architecture(arch, ib::GridFittedVerticalBoundaries) =
    GridFittedVerticalBoundaries(on_architecture(arch, ib.bottom_height),
                                 on_architecture(arch, ib.top_height),
                                 ib.immersed_condition)

Adapt.adapt_structure(to, ib::GridFittedVerticalBoundaries) =
    GridFittedVerticalBoundaries(adapt(to, ib.bottom_height),
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
function materialize_immersed_boundary(grid, ib::GridFittedVerticalBoundaries)
    domain_bottom, domain_top = z_domain(grid)
    bottom = materialize_height(grid, ib.bottom_height, domain_bottom)
    top    = materialize_height(grid, ib.top_height, domain_top)

    compute_ib = GridFittedVerticalBoundaries(bottom, top, ib.immersed_condition)

    @apply_regionally compute_numerical_heights!(bottom, top, grid, compute_ib)
    fill_halo_regions!(bottom)
    fill_halo_regions!(top)

    return GridFittedVerticalBoundaries(height_data(bottom), height_data(top), ib.immersed_condition)
end

# A missing bottom or top is the bottom or top of the domain
materialize_height(grid, ::Nothing, domain_boundary) = ConstantField(domain_boundary)

# Only a `ConstantField` on the domain boundary dispatches as a missing boundary; any other constant is materialized
materialize_height(grid, height::ConstantField, domain_boundary) =
    height.constant == domain_boundary ? height : materialize_height(grid, height.constant, domain_boundary)

function materialize_height(grid, height, domain_boundary)
    height_field = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(height_field, height)
    return height_field
end

height_data(height_field) = height_field.data
height_data(height::ConstantField) = height

compute_numerical_heights!(bottom, top, grid, ib) =
    launch!(architecture(grid), grid, :xy, _compute_numerical_heights!, bottom, top, grid, ib)

@kernel function _compute_numerical_heights!(bottom, top, grid, ib::GridFittedVerticalBoundaries)
    i, j = @index(Global, NTuple)
    zᵇ = @inbounds bottom[i, j, 1]
    zᵗ = @inbounds top[i, j, 1]
    ẑᵇ = rnode(i, j, 1, grid, c, c, f)
    ẑᵗ = rnode(i, j, grid.Nz+1, grid, c, c, f)
    center_condition = ib.immersed_condition isa CenterImmersedCondition
    for k in 1:grid.Nz
        z⁻ = rnode(i, j, k,   grid, c, c, f)
        z⁺ = rnode(i, j, k+1, grid, c, c, f)
        z  = rnode(i, j, k,   grid, c, c, c)
        immersed_by_bottom = ifelse(center_condition, z ≤ zᵇ, z⁺ ≤ zᵇ)
        immersed_by_top    = ifelse(center_condition, z ≥ zᵗ, z⁻ ≥ zᵗ)
        ẑᵇ = ifelse(immersed_by_bottom, z⁺, ẑᵇ)
        ẑᵗ = ifelse(immersed_by_top, min(ẑᵗ, z⁻), ẑᵗ)
    end

    closed = ẑᵗ ≤ ẑᵇ
    store_column_heights!(bottom, top, i, j, grid, ẑᵇ, ẑᵗ, closed)
end

# Closed columns collapse onto the top of the domain, or onto the bottom if the bottom is not stored
@inline function store_column_heights!(bottom, top, i, j, grid, ẑᵇ, ẑᵗ, closed)
    closed_height = closed_column_height(i, j, grid, bottom)
    store_height!(bottom, i, j, ifelse(closed, closed_height, ẑᵇ))
    store_height!(top,    i, j, ifelse(closed, closed_height, ẑᵗ))
    return nothing
end

@inline closed_column_height(i, j, grid, bottom) = rnode(i, j, grid.Nz+1, grid, c, c, f)
@inline closed_column_height(i, j, grid, bottom::ConstantField) = bottom.constant

@inline store_height!(height, i, j, ẑ) = @inbounds height[i, j, 1] = ẑ
@inline store_height!(::ConstantField, i, j, ẑ) = nothing

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom)
    # We use `rnode` for the `immersed_cell` because we do not want to have
    # wetting or drying that could happen for a moving grid if we use znode
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zb = @inbounds ib.bottom_height[i, j, 1]
    return z ≤ zb
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedVerticalBoundaries)
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zb = @inbounds ib.bottom_height[i, j, 1]
    zt = @inbounds ib.top_height[i, j, 1]
    return (z ≤ zb) | (z ≥ zt)
end

@inline function _immersed_cell(i, j, k::AbstractArray, underlying_grid, ib::GridFittedBottom)
    # We use `rnode` for the `immersed_cell` because we do not want to have
    # wetting or drying that could happen for a moving grid if we use znode
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zb = @inbounds ib.bottom_height[i, j, 1]
    _zb = Base.stack(collect(zb for _ in k))
    return z .≤ _zb
end

@inline function _immersed_cell(i, j, k::AbstractArray, underlying_grid, ib::GridFittedVerticalBoundaries)
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

@inline static_column_depthᶜᶜᵃ(i, j, ibg::AGFBIBG) = @inbounds ibg.immersed_boundary.top_height[i, j, 1] - ibg.immersed_boundary.bottom_height[i, j, 1]
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

function Base.:(==)(gfb1::GridFittedVerticalBoundaries, gfb2::GridFittedVerticalBoundaries)
    return bottom_heights_equal(gfb1.bottom_height, gfb2.bottom_height) &&
           bottom_heights_equal(gfb1.top_height, gfb2.top_height) &&
           gfb1.immersed_condition == gfb2.immersed_condition
end
