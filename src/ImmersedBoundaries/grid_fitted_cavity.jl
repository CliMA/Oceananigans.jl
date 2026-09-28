using Oceananigans.Grids: Grids, constructor_arguments, rnode
using Oceananigans.Fields: AbstractField, Field, fill_halo_regions!, interior

struct GridFittedCavity{H, C, L} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    ceiling_height :: C
    ice_load :: L
end

const GFCIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedCavity}

"""
$(TYPEDSIGNATURES)

Return an immersed boundary in which the fluid occupies the cavity between
`bottom_height` and `ceiling_height`, as in the ocean cavity beneath an ice shelf
whose draft is `ceiling_height`.

Arguments
=========

* `bottom_height`: an array or function of `(x, y)` giving the bottom height in absolute `z`.

* `ceiling_height`: an array or function of `(x, y)` giving the height of the underside of the
                    ceiling (e.g. the ice-shelf draft) in absolute `z`. Columns where
                    `ceiling_height` is at or above the top of the grid are open.

Keyword arguments
=================

* `ice_load`: the potential of the weight of the ice, added to the hydrostatic pressure of every
              column. Either `nothing` (default, no load), a
              [`CavityLoad`](@ref) computed from a reference state, or an array, function of `(x, y)`
              or two-dimensional field such as the one returned by [`cavity_load_potential`](@ref).

A cell is immersed when its center lies below `bottom_height` or above
`ceiling_height`. Columns beneath the ceiling that contain fewer than two wet
cells (in particular columns where `ceiling_height ≤ bottom_height`) are closed
entirely.

Example
=======

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 8, 10), x=(0, 100), y=(0, 100), z=(-100, 0), topology=(Periodic, Periodic, Bounded));

julia> ibg = ImmersedBoundaryGrid(grid, GridFittedCavity(-100, (x, y) -> -20 - 0.2y))
2×8×10 ImmersedBoundaryGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo:
├── immersed_boundary: GridFittedCavity(mean(zb)=-100.0, mean(zd)=-30.0)
├── underlying_grid: 2×8×10 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── Periodic x ∈ [0.0, 100.0)  regularly spaced with Δx=50.0
├── Periodic y ∈ [0.0, 100.0)  regularly spaced with Δy=12.5
└── Bounded  z ∈ [-100.0, 0.0] regularly spaced with Δz=10.0
```
"""
GridFittedCavity(bottom_height, ceiling_height; ice_load=nothing) =
    GridFittedCavity(bottom_height, ceiling_height, ice_load)

function Base.summary(ib::GridFittedCavity)
    bottom_interior  = bottom_height_interior(ib.bottom_height)
    ceiling_interior = bottom_height_interior(ib.ceiling_height)
    zbmean = sum(bottom_interior) / length(bottom_interior)
    zdmean = sum(ceiling_interior) / length(ceiling_interior)
    return string("GridFittedCavity(mean(zb)=", prettysummary(zbmean),
                  ", mean(zd)=", prettysummary(zdmean), ")")
end

Base.summary(ib::GridFittedCavity{<:Function}) = "GridFittedCavity"

function Base.show(io::IO, ib::GridFittedCavity)
    print(io, summary(ib), '\n')
    print(io, "├── bottom_height: ", prettysummary(ib.bottom_height), '\n')
    print(io, "├── ceiling_height: ", prettysummary(ib.ceiling_height), '\n')
    print(io, "└── ice_load: ", prettysummary(ib.ice_load), '\n')
    return nothing
end

Architectures.on_architecture(arch, ib::GridFittedCavity) =
    GridFittedCavity(on_architecture(arch, ib.bottom_height),
                     on_architecture(arch, ib.ceiling_height),
                     on_architecture(arch, ib.ice_load))

Adapt.adapt_structure(to, ib::GridFittedCavity) =
    GridFittedCavity(adapt(to, ib.bottom_height), adapt(to, ib.ceiling_height), adapt(to, ib.ice_load))

struct CavityLoad{B, R}
    buoyancy :: B
    reference_tracers :: R
end

"""
$(TYPEDSIGNATURES)

Return an `ice_load` for a [`GridFittedCavity`](@ref) or [`PartialCellCavity`](@ref) that is
computed by [`cavity_load_potential`](@ref) from `buoyancy` and `reference_tracers` when the
`ImmersedBoundaryGrid` is built. A cavity whose stratification equals `reference_tracers` is then at rest.

Example
=======

```jldoctest
julia> using Oceananigans

julia> underlying_grid = RectilinearGrid(size=(1, 4, 10), x=(0, 1), y=(0, 4), z=(-100, 0), topology=(Periodic, Bounded, Bounded));

julia> ceiling_height(x, y) = y < 2 ? -40 : 0;

julia> ice_load = CavityLoad(BuoyancyTracer(), (; b = (x, y, z) -> 1e-5 * z));

julia> grid = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-100, ceiling_height; ice_load));

julia> grid.immersed_boundary.ice_load[1, 1:4, 1]
4-element Vector{Float64}:
 0.008
 0.008
 0.0
 0.0
```
"""
CavityLoad(buoyancy, reference_tracers::NamedTuple) = CavityLoad{typeof(buoyancy), typeof(reference_tracers)}(buoyancy, reference_tracers)

Base.summary(load::CavityLoad) = string("CavityLoad(", summary(load.buoyancy), ", reference_tracers=", keys(load.reference_tracers), ")")

materialize_ice_load(grid, unloaded_ib, ::Nothing) = nothing

function materialize_ice_load(grid, unloaded_ib, ice_load)
    ice_load_field = Field{Center, Center, Nothing}(grid)
    set_ice_load!(ice_load_field, ice_load)
    fill_halo_regions!(ice_load_field)
    return ice_load_field.data
end

set_ice_load!(ice_load_field, ice_load) = set_bottom_height!(ice_load_field, ice_load)
set_ice_load!(ice_load_field, ice_load::AbstractField) = copyto!(interior(ice_load_field), interior(ice_load))

function materialize_immersed_boundary(grid, ib::GridFittedCavity)
    bottom_field  = Field{Center, Center, Nothing}(grid)
    ceiling_field = Field{Center, Center, Nothing}(grid)
    set_bottom_height!(bottom_field, ib.bottom_height)
    set_bottom_height!(ceiling_field, ib.ceiling_height)
    @apply_regionally compute_numerical_cavity_heights!(bottom_field, ceiling_field, grid)
    fill_halo_regions!(bottom_field)
    fill_halo_regions!(ceiling_field)
    unloaded_ib = GridFittedCavity(bottom_field.data, ceiling_field.data, nothing)
    ice_load = materialize_ice_load(grid, unloaded_ib, ib.ice_load)
    return GridFittedCavity(bottom_field.data, ceiling_field.data, ice_load)
end

compute_numerical_cavity_heights!(bottom_field, ceiling_field, grid) =
    launch!(architecture(grid), grid, :xy, _compute_numerical_cavity_heights!, bottom_field, ceiling_field, grid)

@kernel function _compute_numerical_cavity_heights!(bottom_field, ceiling_field, grid)
    i, j = @index(Global, NTuple)
    zᵇ = @inbounds bottom_field[i, j, 1]
    zᵈ = @inbounds ceiling_field[i, j, 1]

    domain_bottom = rnode(i, j, 1, grid, c, c, f)
    domain_top    = rnode(i, j, grid.Nz+1, grid, c, c, f)

    # Snap to the faces of the outermost cells whose centers are immersed
    ẑᵇ = domain_bottom
    ẑᵈ = domain_top
    for k in 1:grid.Nz
        z⁻ = rnode(i, j, k,   grid, c, c, f)
        z⁺ = rnode(i, j, k+1, grid, c, c, f)
        z  = rnode(i, j, k,   grid, c, c, c)
        ẑᵇ = ifelse(z ≤ zᵇ, z⁺, ẑᵇ)
        ẑᵈ = ifelse(z ≥ zᵈ, min(ẑᵈ, z⁻), ẑᵈ)
    end

    Nʷ = 0
    for k in 1:grid.Nz
        z = rnode(i, j, k, grid, c, c, c)
        Nʷ += ifelse((ẑᵇ < z) & (z < ẑᵈ), 1, 0)
    end

    # Columns without a ceiling keep GridFittedBottom's single-cell behavior
    close_column = (ẑᵈ < domain_top) & (Nʷ < 2)
    ẑᵇ = ifelse(close_column, domain_top, ẑᵇ)
    ẑᵈ = ifelse(close_column, domain_top, ẑᵈ)

    @inbounds bottom_field[i, j, 1]  = ẑᵇ
    @inbounds ceiling_field[i, j, 1] = ẑᵈ
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedCavity)
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zᵇ = @inbounds ib.bottom_height[i, j, 1]
    zᵈ = @inbounds ib.ceiling_height[i, j, 1]
    return (z ≤ zᵇ) | (z ≥ zᵈ)
end

@inline function _immersed_cell(i, j, k::AbstractArray, underlying_grid, ib::GridFittedCavity)
    z  = rnode(i, j, k, underlying_grid, c, c, c)
    zᵇ = Base.stack(collect(@inbounds(ib.bottom_height[i, j, 1]) for _ in k))
    zᵈ = Base.stack(collect(@inbounds(ib.ceiling_height[i, j, 1]) for _ in k))
    return (z .≤ zᵇ) .| (z .≥ zᵈ)
end

@inline static_column_depthᶜᶜᵃ(i, j, ibg::GFCIBG) =
    @inbounds ibg.immersed_boundary.ceiling_height[i, j, 1] - ibg.immersed_boundary.bottom_height[i, j, 1]

@inline static_column_depthᶠᶜᵃ(i, j, ibg::GFCIBG) = active_column_depthᶠᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::GFCIBG) = active_column_depthᶜᶠᵃ(i, j, ibg)

const XFlatGFCIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:GridFittedCavity}
const YFlatGFCIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:GridFittedCavity}

# Disambiguate against the generic XFlatAGFIBG/YFlatAGFIBG methods in grid_fitted_bottom.jl.
@inline static_column_depthᶠᶜᵃ(i, j, ibg::XFlatGFCIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::YFlatGFCIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)

function Grids.constructor_arguments(grid::GFCIBG)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid.underlying_grid)
    grid_fitted_cavity_args = Dict(:bottom_height  => grid.immersed_boundary.bottom_height,
                                   :ceiling_height => grid.immersed_boundary.ceiling_height,
                                   :ice_load       => grid.immersed_boundary.ice_load)
    return underlying_grid_args, underlying_grid_kwargs, grid_fitted_cavity_args
end

function Base.:(==)(gfc1::GridFittedCavity, gfc2::GridFittedCavity)
    return bottom_heights_equal(gfc1.bottom_height, gfc2.bottom_height) &&
           bottom_heights_equal(gfc1.ceiling_height, gfc2.ceiling_height) &&
           bottom_heights_equal(gfc1.ice_load, gfc2.ice_load)
end
