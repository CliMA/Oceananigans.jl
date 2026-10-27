using Oceananigans.Grids: Grids, constructor_arguments, rnode
using Oceananigans.Fields: Field, fill_halo_regions!, interior
using Oceananigans.BoundaryConditions: FBC
using Oceananigans.Utils: KernelParameters
using OffsetArrays: OffsetArray

#####
##### GridFittedVerticalBoundaries (2.5D immersed boundary with modified bottom and top heights)
#####

abstract type AbstractGridFittedBottom{H} <: AbstractGridFittedBoundary end

# To enable comparison with PartialCellBottom in the limiting case that
# fractional cell height is 1.0.
struct CenterImmersedCondition end
struct InterfaceImmersedCondition end

struct GridFittedVerticalBoundaries{H, T, I, KB, KT} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    top_height :: T
    immersed_condition :: I
    bottom_cell_index :: KB
    top_cell_index :: KT
end

GridFittedVerticalBoundaries(bottom_height, top_height, immersed_condition) =
    GridFittedVerticalBoundaries(bottom_height, top_height, immersed_condition, nothing, nothing)

const GridFittedBottom{H, I} = GridFittedVerticalBoundaries{H, Nothing, I, <:Any, Nothing}
const GridFittedWithTop = GridFittedVerticalBoundaries{<:Any, <:Any, <:Any, <:Any, <:AbstractArray}

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
    GridFittedBottom(bottom_height, [immersed_condition=CenterImmersedCondition()])

Return a [`GridFittedVerticalBoundaries`](@ref) without an immersed top, bounded below by `bottom_height`.
"""
function GridFittedBottom(bottom_height, immersed_condition=CenterImmersedCondition())
    return GridFittedVerticalBoundaries(bottom_height, nothing, immersed_condition)
end

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

function set_bottom_height!(bottom_field, bottom_height::OffsetArray)
    source = on_architecture(architecture(bottom_field), bottom_height_interior(bottom_height))
    copyto!(interior(bottom_field), source)
    return bottom_field
end

height_field(height_data, grid) = Field{Center, Center, Nothing}(grid; data=height_data)

function height_field(height::Number, grid)
    field = Field{Center, Center, Nothing}(grid)
    fill!(field, height)
    return field
end

"""
$(TYPEDSIGNATURES)

Return a `Field` at `(Center, Center, Nothing)` that wraps the bottom height of `grid`, which is
stored on `grid.immersed_boundary` as a bare `OffsetArray`. The returned `Field` shares its data
with `grid`, so mutating it mutates the bottom height of `grid`. If `grid` has no immersed bottom,
the returned `Field` is a copy filled with the bottom of the domain.
"""
bottom_height_field(grid::IBG) = height_field(something(grid.immersed_boundary.bottom_height, first(z_domain(grid))), grid.underlying_grid)

"""
$(TYPEDSIGNATURES)

Return a `Field` at `(Center, Center, Nothing)` that wraps the top height of `grid`. The returned
`Field` shares its data with `grid`. If `grid` has no immersed top, the returned `Field` is a copy
filled with the top of the domain.
"""
top_height_field(grid::IBG) = height_field(something(grid.immersed_boundary.top_height, last(z_domain(grid))), grid.underlying_grid)

function height_statistics(height)
    height_interior = bottom_height_interior(height)
    mean_height = sum(height_interior) / length(height_interior)
    return mean_height, minimum(height_interior), maximum(height_interior)
end

top_height_summary(::Nothing) = ""

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
                                 ib.immersed_condition,
                                 on_architecture(arch, ib.bottom_cell_index),
                                 on_architecture(arch, ib.top_cell_index))

# Kernels read the bottom and top through `bottom_cell_index` and `top_cell_index` only, see `bottom_height(i, j, ibg)`
Adapt.adapt_structure(to, ib::GridFittedVerticalBoundaries) =
    GridFittedVerticalBoundaries(nothing,
                                 nothing,
                                 adapt(to, ib.immersed_condition),
                                 adapt(to, ib.bottom_cell_index),
                                 adapt(to, ib.top_cell_index))

"""
$(TYPEDSIGNATURES)

Returns a new `ib` that holds the numerical `immersed_boundary`.
If `ib` is an `AbstractGridFittedBottom`, `ib.bottom_height` is an `OffsetArray` holding the
z-coordinate of the top-most interface of the last ``immersed`` cell in the column (wrap it as a
`Field` with [`bottom_height_field`](@ref)). If `ib` is a `GridFittedBoundary`, `ib.mask` is a `Field` of
booleans that indicates whether a cell is immersed or not.
"""
function materialize_immersed_boundary(grid, ib::GridFittedVerticalBoundaries)
    bottom = materialize_height(grid, ib.bottom_height)
    top    = materialize_height(grid, ib.top_height)

    compute_ib = GridFittedVerticalBoundaries(bottom, top, ib.immersed_condition)

    @apply_regionally compute_numerical_heights!(bottom, top, grid, compute_ib)

    bottom_cell_index = index_field(grid, bottom, index_type(grid))
    top_cell_index    = index_field(grid, top,    index_type(grid))

    @apply_regionally compute_cell_indices!(grid, compute_ib, bottom_cell_index, top_cell_index, bottom, top)

    return GridFittedVerticalBoundaries(stored_data(bottom), 
                                        stored_data(top), 
                                        ib.immersed_condition,
                                        stored_data(bottom_cell_index), 
                                        stored_data(top_cell_index))
end

# Without a top, kernels take the top of the domain and need no `top_cell_index`
index_field(grid, top, FT) = Field{Center, Center, Nothing}(grid, FT)
index_field(grid, ::Nothing, FT) = nothing

stored_data(field) = field.data
stored_data(::Nothing) = nothing

materialize_height(grid, ::Nothing) = nothing

function materialize_height(grid, height)
    height_field = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(height_field, height)
    fill_halo_regions!(height_field)
    return height_field
end

# A missing bottom or top is the bottom or top of the domain
@inline height_value(i, j, height,    missing_height) = @inbounds height[i, j, 1]
@inline height_value(i, j, ::Nothing, missing_height) = missing_height

# Over the halos too, so that the heights that the halos interpolate are fitted to the vertical grid as well
function compute_numerical_heights!(bottom, top, grid, ib)
    parameters = height_kernel_parameters(bottom, top)
    return launch!(architecture(grid), grid, parameters, _compute_numerical_heights!, bottom, top, grid, ib)
end

height_kernel_parameters(bottom, top) = KernelParameters(axes(bottom.data, 1), axes(bottom.data, 2))
height_kernel_parameters(::Nothing, top) = KernelParameters(axes(top.data, 1), axes(top.data, 2))
height_kernel_parameters(::Nothing, ::Nothing) = :xy

@kernel function _compute_numerical_heights!(bottom, top, grid, ib::GridFittedVerticalBoundaries)
    i, j = @index(Global, NTuple)
    ẑᵇ = rnode(i, j, 1, grid, c, c, f)
    ẑᵗ = rnode(i, j, size(grid, 3)+1, grid, c, c, f)
    zᵇ = height_value(i, j, bottom, ẑᵇ)
    zᵗ = height_value(i, j, top,    ẑᵗ)
    center_condition = ib.immersed_condition isa CenterImmersedCondition
    for k in 1:size(grid, 3)
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
    store_column_value!(bottom, i, j, ifelse(closed, closed_height, ẑᵇ))
    store_column_value!(top,    i, j, ifelse(closed, closed_height, ẑᵗ))
    return nothing
end

@inline closed_column_height(i, j, grid, bottom)    = rnode(i, j, size(grid, 3)+1, grid, c, c, f)
@inline closed_column_height(i, j, grid, ::Nothing) = rnode(i, j, 1, grid, c, c, f)

@inline store_column_value!(field, i, j, value)     = @inbounds field[i, j, 1] = value
@inline store_column_value!(::Nothing, i, j, value) = nothing

# The cell indices are computed from the heights in the halos as well
function compute_cell_indices!(grid, ib, bottom_index, top_index, bottom, top)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    parameters = KernelParameters(1-Hx:Nx+Hx, 1-Hy:Ny+Hy)
    return launch!(architecture(grid), grid, parameters, _compute_cell_indices!, bottom_index, top_index, grid, ib, bottom, top)
end

# kᵇ (kᵗ) is the first (last) active cell of the column; kᵇ = Nz + 1 and kᵗ = Nz in closed columns, kᵗ = 0 without a stored bottom
@kernel function _compute_cell_indices!(bottom_cell_index, top_cell_index, grid, ib::GridFittedVerticalBoundaries, bottom, top)
    i, j = @index(Global, NTuple)
    rᵇ = height_value(i, j, bottom, rnode(i, j, 1, grid, c, c, f))
    rᵗ = height_value(i, j, top,    rnode(i, j, size(grid, 3)+1, grid, c, c, f))
    kᵇ = size(grid, 3) + 1
    kᵗ = 0

    for k in size(grid, 3):-1:1
        r = rnode(i, j, k, grid, c, c, c)
        kᵇ = ifelse(r ≤ rᵇ, kᵇ, k)
    end

    for k in 1:size(grid, 3)
        r = rnode(i, j, k, grid, c, c, c)
        kᵗ = ifelse(r < rᵗ, k, kᵗ)
    end

    store_column_value!(bottom_cell_index, i, j, kᵇ)
    store_column_value!(top_cell_index,    i, j, kᵗ)
end

@inline bottom_cell(i, j, grid, ib::GridFittedVerticalBoundaries) = height_value(i, j, ib.bottom_cell_index, 1)
@inline    top_cell(i, j, grid, ib::GridFittedVerticalBoundaries) = height_value(i, j, ib.top_cell_index, size(grid, 3))

@inline function bottom_height(i, j, ibg::GFBIBG)
    kᵇ = bottom_cell(i, j, ibg.underlying_grid, ibg.immersed_boundary)
    return rnode(i, j, kᵇ, ibg.underlying_grid, c, c, f)
end

# AbstractGridFittedBottomImmersedBoundaryGrid
const AGFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}

@inline function top_height(i, j, ibg::AGFBIBG)
    kᵗ = top_cell(i, j, ibg.underlying_grid, ibg.immersed_boundary)
    return rnode(i, j, kᵗ+1, ibg.underlying_grid, c, c, f)
end

@inline _immersed_cell(i, j, k, underlying_grid, ib::AbstractGridFittedBottom) = k < bottom_cell(i, j, underlying_grid, ib)
@inline _immersed_cell(i, j, k, underlying_grid, ib::GridFittedWithTop) = (k < bottom_cell(i, j, underlying_grid, ib)) | (k > top_cell(i, j, underlying_grid, ib))

@inline function _immersed_cell(i, j, k::AbstractArray, underlying_grid, ib::GridFittedBottom)
    # We use `rnode` for the `immersed_cell` because we do not want to have
    # wetting or drying that could happen for a moving grid if we use znode
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zb = @inbounds ib.bottom_height[i, j, 1]
    _zb = Base.stack(collect(zb for _ in k))
    return z .≤ _zb
end

@inline function _immersed_cell(i, j, k::AbstractArray, underlying_grid, ib::GridFittedWithTop)
    kᵇ = bottom_cell(i, j, underlying_grid, ib)
    kᵗ = top_cell(i, j, underlying_grid, ib)
    return (k .< kᵇ) .| (k .> kᵗ)
end

#####
##### Static column depth
#####

@inline static_column_depthᶜᶜᵃ(i, j, ibg::AGFBIBG) = top_height(i, j, ibg) - bottom_height(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::AGFBIBG) = face_column_depth(i, j, i, j-1, ibg, ibg.immersed_boundary)
@inline static_column_depthᶠᶜᵃ(i, j, ibg::AGFBIBG) = face_column_depth(i, j, i-1, j, ibg, ibg.immersed_boundary)
@inline static_column_depthᶠᶠᵃ(i, j, ibg::AGFBIBG) = min(static_column_depthᶠᶜᵃ(i, j-1, ibg), static_column_depthᶠᶜᵃ(i, j, ibg))

# Without a top, adjacent columns share their top, so the shallower column depth is the open depth of the face
@inline face_column_depth(i, j, i⁻, j⁻, ibg, ::GridFittedBottom) = min(static_column_depthᶜᶜᵃ(i⁻, j⁻, ibg), static_column_depthᶜᶜᵃ(i, j, ibg))

# Beneath a top only the overlap of columns (i⁻, j⁻) and (i, j) is open. 
@inline function face_column_depth(i, j, i⁻, j⁻, ibg, ib)
    ug = ibg.underlying_grid
    k⁻ = max(bottom_cell(i⁻, j⁻, ug, ib), bottom_cell(i, j, ug, ib))
    k⁺ = min(   top_cell(i⁻, j⁻, ug, ib),    top_cell(i, j, ug, ib))
    H  = rnode(i, j, k⁺+1, ug, c, c, f) - rnode(i, j, k⁻, ug, c, c, f)

    bottom_deficit = Δrᶜᶜᶜ(i, j, k⁻, ug) - min(Δrᶜᶜᶜ(i⁻, j⁻, k⁻, ibg), Δrᶜᶜᶜ(i, j, k⁻, ibg))
    top_deficit    = Δrᶜᶜᶜ(i, j, k⁺, ug) - min(Δrᶜᶜᶜ(i⁻, j⁻, k⁺, ibg), Δrᶜᶜᶜ(i, j, k⁺, ibg))

    return max(H - bottom_deficit - ifelse(k⁺ > k⁻, top_deficit, zero(H)), zero(H))
end

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
