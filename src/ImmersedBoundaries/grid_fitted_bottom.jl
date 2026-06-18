using Oceananigans.Grids: Grids, constructor_arguments, rnode
using Oceananigans.Fields: Field, fill_halo_regions!
using Oceananigans.BoundaryConditions: FBC

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

Arguments
=========

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
    print(io, "└── bottom_height: ", prettysummary(ib.bottom_height), '\n')
end

Architectures.on_architecture(arch, ib::GridFittedBottom) = GridFittedBottom(on_architecture(arch, ib.bottom_height), ib.immersed_condition)

function Architectures.on_architecture(arch, ib::GridFittedBottom{<:Field})
    architecture(ib.bottom_height) == arch && return ib
    arch_grid = on_architecture(arch, ib.bottom_height.grid)
    new_bottom_height = Field{Center, Center, Nothing}(arch_grid)
    set!(new_bottom_height, ib.bottom_height)
    fill_halo_regions!(new_bottom_height)
    return GridFittedBottom(new_bottom_height, ib.immersed_condition)
end

Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.bottom_height), adapt(to, ib.immersed_condition))

"""
$(TYPEDSIGNATURES)

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
        immersed_cell = ifelse(condition isa CenterImmersedCondition, z ≤ zb, z⁺ ≤ zb)
        @inbounds bottom_field[i, j, 1] = ifelse(immersed_cell, z⁺, bottom_field[i, j, 1])
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
    _zb = Base.stack(collect(zb for _ in k))
    return z .≤ _zb
end

# Multi-envelope grids: `rnode` is the *uniform reference* coordinate, which on a terrain-following grid is not
# the physical depth, so masking `rnode ≤ bottom_height` (a physical depth) carves topography at the wrong
# place. Instead mask against the *resting* (σ_fs = 1) physical depth of the cell centre — the envelope ẑ(r),
# measured from the surface as the summed physical thickness Δr·σᵉ of the cells above. This is static (so it
# neither wets nor dries as the grid breathes, like `rnode`) yet lives in physical space (so it matches
# `bottom_height`, unlike `rnode`). Precomputed once into `zᶜᶜᶜᵉ` (by `compute_resting_znodeᶜᶜᶜ!`) so this
# masking-hot lookup is O(1) — the column sum would be O(Nz) every step for every cell.
@inline resting_znodeᶜᶜᶜ(i, j, k, grid) = @inbounds grid.z.zᶜᶜᶜᵉ[i, j, k]

@inline function _immersed_cell(i, j, k, underlying_grid::MultiEnvelopeGrid, ib::GridFittedBottom)
    z  = resting_znodeᶜᶜᶜ(i, j, k, underlying_grid)
    zb = @inbounds ib.bottom_height[i, j, 1]
    return z ≤ zb
end

# resting (σ_fs = 1) physical depth of the *face* below cell k: −Σ_{k′≥k} Δr·σᵉ (the envelope ẑ at that face)
@inline function resting_znodeᶜᶜᶠ(i, j, k, grid)
    z = grid.z
    depth = zero(eltype(grid))
    @inbounds for k′ in k:grid.Nz
        depth += (z.Δᵃᵃᶜ isa Number ? z.Δᵃᵃᶜ : z.Δᵃᵃᶜ[k′]) * z.σᶜᶜᵉ[i, j, k′]
    end
    return -depth
end

# Bottom-height snapping must also use the resting physical znode (not rnode), or it snaps the bottom to a
# reference level — e.g. −300 m → −280 m on a terrain-following column — which then masks the whole bottom zone.
compute_numerical_bottom_height!(bottom_field, grid::MultiEnvelopeGrid, ib) =
    launch!(architecture(grid), grid, :xy, _compute_me_numerical_bottom_height!, bottom_field, grid, ib)

@kernel function _compute_me_numerical_bottom_height!(bottom_field, grid, ib::GridFittedBottom)
    i, j = @index(Global, NTuple)
    zb = @inbounds bottom_field[i, j, 1]
    condition = ib.immersed_condition
    @inbounds bottom_field[i, j, 1] = resting_znodeᶜᶜᶠ(i, j, 1, grid)
    for k in 1:grid.Nz
        z⁺ = resting_znodeᶜᶜᶠ(i, j, k+1, grid)
        z  = resting_znodeᶜᶜᶜ(i, j, k,   grid)
        immersed_cell = ifelse(condition isa CenterImmersedCondition, z ≤ zb, z⁺ ≤ zb)
        @inbounds bottom_field[i, j, 1] = ifelse(immersed_cell, z⁺, bottom_field[i, j, 1])
    end
end

#####
##### Static column depth
#####

# AbstractGridFittedBottomImmersedBoundaryGrid
const AGFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}

@inline static_column_depthᶜᶜᵃ(i, j, ibg::AGFBIBG) = @inbounds rnode(i, j, ibg.Nz+1, ibg, c, c, f) - ibg.immersed_boundary.bottom_height[i, j, 1]
@inline static_column_depthᶜᶠᵃ(i, j, ibg::AGFBIBG) = min(static_column_depthᶜᶜᵃ(i, j-1, ibg), static_column_depthᶜᶜᵃ(i, j, ibg))
@inline static_column_depthᶠᶜᵃ(i, j, ibg::AGFBIBG) = min(static_column_depthᶜᶜᵃ(i-1, j, ibg), static_column_depthᶜᶜᵃ(i, j, ibg))
@inline static_column_depthᶠᶠᵃ(i, j, ibg::AGFBIBG) = min(static_column_depthᶠᶜᵃ(i, j-1, ibg), static_column_depthᶠᶜᵃ(i, j, ibg))

# Multi-envelope immersed grids. The bottom is masked/snapped against the resting PHYSICAL znode (see
# `_compute_me_numerical_bottom_height!`), so the snapped `bottom_height` already IS the physical resting depth
# of the wet column — i.e. −bottom_height = Σ_wet Δr·σᵉ exactly. We therefore look it up in O(1) instead of
# re-summing the wet cells (which, with the O(Nz) resting-znode masking, was O(Nz²) per call → O(Nz³)/column).
const MultiEnvelopeAGFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:MultiEnvelopeGrid, <:AbstractGridFittedBottom}

@inline static_column_depthᶜᶜᵃ(i, j, ibg::MultiEnvelopeAGFBIBG) = @inbounds -ibg.immersed_boundary.bottom_height[i, j, 1]

# Make sure column_height works for horizontally-Flat topologies.
XFlatAGFIBG = ImmersedBoundaryGrid{<:Any, <:Flat, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}
YFlatAGFIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Flat, <:Any, <:Any, <:AbstractGridFittedBottom}

@inline static_column_depthᶠᶜᵃ(i, j, ibg::XFlatAGFIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::YFlatAGFIBG) = static_column_depthᶜᶜᵃ(i, j, ibg)
@inline static_column_depthᶠᶠᵃ(i, j, ibg::XFlatAGFIBG) = static_column_depthᶜᶠᵃ(i, j, ibg)
@inline static_column_depthᶠᶠᵃ(i, j, ibg::YFlatAGFIBG) = static_column_depthᶠᶜᵃ(i, j, ibg)

function Grids.constructor_arguments(grid::AGFBIBG)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid.underlying_grid)
    grid_fitted_bottom_args = Dict(:bottom_height      => grid.immersed_boundary.bottom_height,
                                   :immersed_condition => grid.immersed_boundary.immersed_condition)
    return underlying_grid_args, underlying_grid_kwargs, grid_fitted_bottom_args
end

function Base.:(==)(gfb1::GridFittedBottom, gfb2::GridFittedBottom)
    return gfb1.bottom_height == gfb2.bottom_height && gfb1.immersed_condition == gfb2.immersed_condition
end
