using Oceananigans.Fields: Field, set!
using Oceananigans.Grids: Grids, constructor_arguments
using Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶠᶜᶜ, Δrᶜᶠᶜ

"""
    GridFittedBoundary(mask)

Return a immersed boundary with a three-dimensional `mask`.
"""
struct GridFittedBoundary{M} <: AbstractGridFittedBoundary
    mask :: M
end

@inline _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBoundary{<:AbstractArray}) = @inbounds ib.mask[i, j, k]

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBoundary)
    x, y, z = node(i, j, k, underlying_grid, c, c, c)
    return ib.mask(x, y, z)
end

function compute_mask(grid, ib)
    mask_field = Field{Center, Center, Center}(grid, Bool)
    set!(mask_field, ib.mask)
    fill_halo_regions!(mask_field)
    return mask_field
end

function materialize_immersed_boundary(grid, ib::GridFittedBoundary)
    mask_field = compute_mask(grid, ib)
    return GridFittedBoundary(mask_field)
end

Architectures.on_architecture(arch, ib::GridFittedBoundary{<:Field}) = GridFittedBoundary(compute_mask(on_architecture(arch, ib.mask.grid), ib))
Architectures.on_architecture(arch, ib::GridFittedBoundary) = ib # need a workaround...

Adapt.adapt_structure(to, ib::AbstractGridFittedBoundary) = GridFittedBoundary(adapt(to, ib.mask))

const AGFBoundIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBoundary}
function Grids.constructor_arguments(grid::AGFBoundIBG)
    underlying_grid_args, underlying_grid_kwargs = constructor_arguments(grid.underlying_grid)
    grid_fitted_boundary_args = Dict(:mask => grid.immersed_boundary.mask)
    return underlying_grid_args, underlying_grid_kwargs, grid_fitted_boundary_args
end

Base.:(==)(gfb1::GridFittedBoundary, gfb2::GridFittedBoundary) = gfb1.mask == gfb2.mask

# The generic fallback returns `grid.Lz`, which ignores the mask
@inline function static_column_depthᶜᶜᵃ(i, j, ibg::AGFBoundIBG)
    H = zero(ibg)
    for k in 1:ibg.Nz
        H += ifelse(peripheral_node(i, j, k, ibg, c, c, c), zero(ibg), Δrᶜᶜᶜ(i, j, k, ibg))
    end
    return H
end

@inline static_column_depthᶠᶜᵃ(i, j, ibg::AGFBoundIBG) = active_column_depthᶠᶜᵃ(i, j, ibg)
@inline static_column_depthᶜᶠᵃ(i, j, ibg::AGFBoundIBG) = active_column_depthᶜᶠᵃ(i, j, ibg)
