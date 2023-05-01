using KernelAbstractions: @kernel, @index
using Statistics
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: location, ZReducedField, Field

instantiate(T::Type) = T()
instantiate(t) = t

mask_immersed!(field, grid, loc, value) = nothing
mask_immersed!(field::Field, value=zero(eltype(field.grid))) =
    mask_immersed!(field, field.grid, location(field), value)

"""
    mask_immersed!(field::Field, grid::ImmersedBoundaryGrid, loc, value)

Mask `field` defined on `grid` with a `value` at `loc`ations at grid points where [`peripheral_node`](@ref)
is to `true`.
"""
function mask_immersed!(field::Field, grid::ImmersedBoundaryGrid, loc, value)
    arch = architecture(field)
    loc = instantiate.(loc)
    launch!(arch, grid, :xyz, _mask_immersed!, field, loc, grid, value)
    return nothing
end


@kernel function _mask_immersed!(field, loc, grid, value)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = scalar_mask(i, j, k, grid, grid.immersed_boundary, loc..., value, field)
end

mask_immersed_field_xy!(field,     args...; kw...) = nothing
mask_immersed_field_xy!(::Nothing, args...; kw...) = nothing
mask_immersed_field_xy!(field, value=zero(eltype(field.grid)); k, mask = peripheral_node) =
    mask_immersed_field_xy!(field, field.grid, location(field), value; k, mask)

"""
    mask_immersed_field_xy!(field::Field, grid::ImmersedBoundaryGrid, loc, value; k, mask=peripheral_node)

Mask `field` on `grid` with a `value` on the slices `[:, :, k]` where `mask` is `true`.
"""
function mask_immersed_field_xy!(field::Field, grid::ImmersedBoundaryGrid, loc, value; k, mask)
    arch = architecture(field)
    loc = instantiate.(loc)
    return launch!(arch, grid, :xy,
                   _mask_immersed_field_xy!, field, loc, grid, value, k, mask)
end

@kernel function _mask_immersed_field_xy!(field, loc, grid, value, k, mask)
    i, j = @index(Global, NTuple)
    @inbounds field[i, j, k] = scalar_mask(i, j, k, grid, grid.immersed_boundary, loc..., value, field, mask)
end

#####
##### Masking for GridFittedBoundary
#####

@inline scalar_mask(i, j, k, grid, ::AbstractGridFittedBoundary, LX, LY, LZ, value, field, mask=peripheral_node) =
    @inbounds ifelse(mask(i, j, k, grid, LX, LY, LZ), value, field[i, j, k])
