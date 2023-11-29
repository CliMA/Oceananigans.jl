using OffsetArrays

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

function ImmersedBoundaryGrid(grid, ib::GridFittedBoundary; precompute_mask=true)
    TX, TY, TZ = topology(grid)

    # TODO: validate ib

    if precompute_mask
        mask_field = compute_mask(grid, ib)
        new_ib = GridFittedBoundary(mask_field)
        return ImmersedBoundaryGrid{TX, TY, TZ}(grid, new_ib)
    else
        return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib)
    end
end

on_architecture(arch, ib::GridFittedBoundary{<:Field}) = GridFittedBoundary(compute_mask(on_architecture(arch, ib.mask.grid), ib))
on_architecture(arch, ib::GridFittedBoundary) = ib # need a workaround...

Adapt.adapt_structure(to, ib::AbstractGridFittedBoundary) = GridFittedBoundary(adapt(to, ib.mask))

