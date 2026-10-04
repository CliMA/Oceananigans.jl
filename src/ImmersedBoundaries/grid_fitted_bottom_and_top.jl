using Oceananigans.Grids: Grids, constructor_arguments, rnode
using Oceananigans.Fields: AbstractField, Field, fill_halo_regions!, interior

struct GridFittedBottomAndTop{H, T, L} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    top_height :: T
    top_load :: L
end

const GFBTIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBottomAndTop}

"""
$(TYPEDSIGNATURES)

Return an immersed boundary in which the fluid lies between `bottom_height` and `top_height`,
for example the ocean beneath an ice shelf whose draft is `top_height`.

Arguments
=========

* `bottom_height`: an array or function of `(x, y)` giving the bottom height in absolute `z`.

* `top_height`: an array or function of `(x, y)` giving the height of the underside of the
                solid top (e.g. an ice-shelf draft) in absolute `z`. Columns where
                `top_height` is at or above the top of the grid are open.

Keyword arguments
=================

* `top_load`: the potential of the weight of the solid top, added to the hydrostatic pressure
              of every column. Either `nothing` (default, no load), a [`TopLoad`](@ref)
              computed from a reference state, or an array, function of `(x, y)` or
              two-dimensional field such as the one returned by [`top_load_potential`](@ref).

A cell is immersed when its center lies below `bottom_height` or above
`top_height`. Columns beneath an immersed top that contain fewer than two wet
cells (in particular columns where `top_height ≤ bottom_height`) are closed
entirely.

Example
=======

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 8, 10), x=(0, 100), y=(0, 100), z=(-100, 0), topology=(Periodic, Periodic, Bounded));

julia> ibg = ImmersedBoundaryGrid(grid, GridFittedBottomAndTop(-100, (x, y) -> -20 - 0.2y))
2×8×10 ImmersedBoundaryGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo:
├── immersed_boundary: GridFittedBottomAndTop(mean(zb)=-100.0, mean(zt)=-30.0)
├── underlying_grid: 2×8×10 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── Periodic x ∈ [0.0, 100.0)  regularly spaced with Δx=50.0
├── Periodic y ∈ [0.0, 100.0)  regularly spaced with Δy=12.5
└── Bounded  z ∈ [-100.0, 0.0] regularly spaced with Δz=10.0
```
"""
GridFittedBottomAndTop(bottom_height, top_height; top_load=nothing) =
    GridFittedBottomAndTop(bottom_height, top_height, top_load)

function Base.summary(ib::GridFittedBottomAndTop)
    bottom_interior = bottom_height_interior(ib.bottom_height)
    top_interior    = bottom_height_interior(ib.top_height)
    zbmean = sum(bottom_interior) / length(bottom_interior)
    ztmean = sum(top_interior) / length(top_interior)
    return string("GridFittedBottomAndTop(mean(zb)=", prettysummary(zbmean),
                  ", mean(zt)=", prettysummary(ztmean), ")")
end

Base.summary(ib::GridFittedBottomAndTop{<:Function}) = "GridFittedBottomAndTop"

function Base.show(io::IO, ib::GridFittedBottomAndTop)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "├── top_height: ", prettysummary(ib.top_height), '\n')
    print(io, "└── top_load: ", prettysummary(ib.top_load), '\n')
    return nothing
end

Architectures.on_architecture(arch, ib::GridFittedBottomAndTop) =
    GridFittedBottomAndTop(on_architecture(arch, ib.bottom_height),
                           on_architecture(arch, ib.top_height),
                           on_architecture(arch, ib.top_load))

Adapt.adapt_structure(to, ib::GridFittedBottomAndTop) =
    GridFittedBottomAndTop(adapt(to, ib.bottom_height), adapt(to, ib.top_height), adapt(to, ib.top_load))

struct TopLoad{B, R}
    buoyancy :: B
    reference_tracers :: R
end

"""
$(TYPEDSIGNATURES)

Return a `top_load` for a [`GridFittedBottomAndTop`](@ref) or [`PartialCellBottomAndTop`](@ref) that is
computed by [`top_load_potential`](@ref) from `buoyancy` and `reference_tracers` when the
`ImmersedBoundaryGrid` is built. A fluid whose stratification equals `reference_tracers` is then at rest.

Example
=======

```jldoctest
julia> using Oceananigans

julia> underlying_grid = RectilinearGrid(size=(1, 4, 10), x=(0, 1), y=(0, 4), z=(-100, 0), topology=(Periodic, Bounded, Bounded));

julia> top_height(x, y) = y < 2 ? -40 : 0;

julia> top_load = TopLoad(BuoyancyTracer(), (; b = (x, y, z) -> 1e-5 * z));

julia> grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottomAndTop(-100, top_height; top_load));

julia> grid.immersed_boundary.top_load[1, 1:4, 1]
4-element Vector{Float64}:
 0.008
 0.008
 0.0
 0.0
```
"""
TopLoad(buoyancy, reference_tracers::NamedTuple) = TopLoad{typeof(buoyancy), typeof(reference_tracers)}(buoyancy, reference_tracers)

Base.summary(load::TopLoad) = string("TopLoad(", summary(load.buoyancy), ", reference_tracers=", keys(load.reference_tracers), ")")

materialize_top_load(grid, unloaded_ib, ::Nothing) = nothing

function materialize_top_load(grid, unloaded_ib, top_load)
    top_load_field = Field{Center, Center, Nothing}(grid)
    set_top_load!(top_load_field, top_load)
    fill_halo_regions!(top_load_field)
    return top_load_field.data
end

set_top_load!(top_load_field, top_load) = set_bottom_height!(top_load_field, top_load)
set_top_load!(top_load_field, top_load::AbstractField) = copyto!(interior(top_load_field), interior(top_load))

function materialize_immersed_boundary(grid, ib::GridFittedBottomAndTop)
    bottom_field = Field{Center, Center, Nothing}(grid)
    top_field    = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(bottom_field, ib.bottom_height)
    set_bottom_height!(top_field, ib.top_height)
    @apply_regionally compute_numerical_bottom_and_top_heights!(bottom_field, top_field, grid)
    fill_halo_regions!(bottom_field)
    fill_halo_regions!(top_field)
    unloaded_ib = GridFittedBottomAndTop(bottom_field.data, top_field.data, nothing)
    top_load = materialize_top_load(grid, unloaded_ib, ib.top_load)
    return GridFittedBottomAndTop(bottom_field.data, top_field.data, top_load)
end

compute_numerical_bottom_and_top_heights!(bottom_field, top_field, grid) =
    launch!(architecture(grid), grid, :xy, _compute_numerical_bottom_and_top_heights!, bottom_field, top_field, grid)

@kernel function _compute_numerical_bottom_and_top_heights!(bottom_field, top_field, grid)
    i, j = @index(Global, NTuple)
    zᵇ = @inbounds bottom_field[i, j, 1]
    zᵗ = @inbounds top_field[i, j, 1]

    domain_bottom = rnode(i, j, 1, grid, c, c, f)
    domain_top    = rnode(i, j, grid.Nz+1, grid, c, c, f)

    # Snap to the faces of the outermost cells whose centers are immersed
    ẑᵇ = domain_bottom
    ẑᵗ = domain_top
    for k in 1:grid.Nz
        z⁻ = rnode(i, j, k,   grid, c, c, f)
        z⁺ = rnode(i, j, k+1, grid, c, c, f)
        z  = rnode(i, j, k,   grid, c, c, c)
        ẑᵇ = ifelse(z ≤ zᵇ, z⁺, ẑᵇ)
        ẑᵗ = ifelse(z ≥ zᵗ, min(ẑᵗ, z⁻), ẑᵗ)
    end

    Nʷ = 0
    for k in 1:grid.Nz
        z = rnode(i, j, k, grid, c, c, c)
        Nʷ += ifelse((ẑᵇ < z) & (z < ẑᵗ), 1, 0)
    end

    # Columns without an immersed top keep GridFittedBottom's single-cell behavior
    close_column = (ẑᵗ < domain_top) & (Nʷ < 2)
    ẑᵇ = ifelse(close_column, domain_top, ẑᵇ)
    ẑᵗ = ifelse(close_column, domain_top, ẑᵗ)

    @inbounds bottom_field[i, j, 1] = ẑᵇ
    @inbounds top_field[i, j, 1]    = ẑᵗ
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottomAndTop)
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zᵇ = @inbounds ib.bottom_height[i, j, 1]
    zᵗ = @inbounds ib.top_height[i, j, 1]
    return (z ≤ zᵇ) | (z ≥ zᵗ)
end

@inline function _immersed_cell(i, j, k::AbstractArray, underlying_grid, ib::GridFittedBottomAndTop)
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zᵇ = Base.stack(collect(@inbounds(ib.bottom_height[i, j, 1]) for _ in k))
    zᵗ = Base.stack(collect(@inbounds(ib.top_height[i, j, 1]) for _ in k))
    return (z .≤ zᵇ) .| (z .≥ zᵗ)
end

@inline static_column_depthᶜᶜᵃ(i, j, ibg::GFBTIBG) =
    @inbounds ibg.immersed_boundary.top_height[i, j, 1] - ibg.immersed_boundary.bottom_height[i, j, 1]

@inline static_column_depthᶠᶜᵃ(i, j, ibg::GFBTIBG) = active_column_depthᶠᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::GFBTIBG) = active_column_depthᶜᶠᵃ(i, j, ibg)

const XFlatGFBTIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:GridFittedBottomAndTop}
const YFlatGFBTIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:GridFittedBottomAndTop}

# Disambiguate against the generic XFlatAGFIBG/YFlatAGFIBG methods in grid_fitted_bottom.jl.
@inline static_column_depthᶠᶜᵃ(i, j, ibg::XFlatGFBTIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::YFlatGFBTIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)

function Grids.constructor_arguments(grid::GFBTIBG)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid.underlying_grid)
    grid_fitted_bottom_and_top_args = Dict(:bottom_height => grid.immersed_boundary.bottom_height,
                                           :top_height    => grid.immersed_boundary.top_height,
                                           :top_load      => grid.immersed_boundary.top_load)
    return underlying_grid_args, underlying_grid_kwargs, grid_fitted_bottom_and_top_args
end

function Base.:(==)(gfc1::GridFittedBottomAndTop, gfc2::GridFittedBottomAndTop)
    return bottom_heights_equal(gfc1.bottom_height, gfc2.bottom_height) &&
           bottom_heights_equal(gfc1.top_height, gfc2.top_height) &&
           bottom_heights_equal(gfc1.top_load, gfc2.top_load)
end
