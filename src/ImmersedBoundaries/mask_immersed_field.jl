using KernelAbstractions: @kernel, @index
using Statistics
using Oceananigans.AbstractOperations: BinaryOperation
using Oceananigans.Fields: location, ZReducedField, Field

instantiate(T::Type) = T()
instantiate(t) = t

mask_immersed_field!(field, grid, loc, value) = nothing
mask_immersed_field!(field::Field, value=zero(eltype(field.grid))) =
    mask_immersed_field!(field, field.grid, location(field), value)

mask_immersed_field!(::Number, args...) = nothing

function mask_immersed_field!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(+)}, value=zero(eltype(bop)))
    a_value = ifelse(bop.b isa Number, -bop.b, value)
    mask_immersed_field!(bop.a, )

    b_value = ifelse(bop.a isa Number, -bop.a, value)
    mask_immersed_field!(bop.b, b_value)
    return nothing
end

function mask_immersed_field!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(-)}, value=zero(eltype(bop)))
    a_value = ifelse(bop.b isa Number, bop.b, value)
    mask_immersed_field!(bop.a, a_value)

    b_value = ifelse(bop.a isa Number, bop.a, value)
    mask_immersed_field!(bop.b, b_value)
    return nothing
end

function mask_immersed_field!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(*)}, value=zero(eltype(bop)))
    mask_immersed_field!(bop.a, value)
    isa(bop.a, Number) && mask_immersed_field!(bop.b, value)
    return nothing
end

function mask_immersed_field!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(/)}, value=zero(eltype(bop)))
    mask_immersed_field!(bop.a, value)
    isa(bop.a, Number) && mask_immersed_field!(bop.b, Inf)
    return nothing
end

function mask_immersed_field!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(^)}, value=zero(eltype(bop)))
    mask_immersed_field!(bop.a, value)
    mask_immersed_field!(bop.b, one(eltype(bop))) # if `a` is a number then the resulting field is not going to be zero
    return nothing
end

"""
    mask_immersed_field!(field::Field, grid::ImmersedBoundaryGrid, loc, value)

masks `field` defined on `grid` with a value `val` at locations where `peripheral_node` evaluates to `true`
"""
function mask_immersed_field!(field::Field, grid::ImmersedBoundaryGrid, loc, value)
    arch = architecture(field)
    loc = instantiate.(loc)
    launch!(arch, grid, :xyz, _mask_immersed_field!, field, loc, grid, value)
    return nothing
end

@kernel function _mask_immersed_field!(field, loc, grid, value)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = scalar_mask(i, j, k, grid, grid.immersed_boundary, loc..., value, field)
end

mask_immersed_field_xy!(field,     args...; kw...) = nothing
mask_immersed_field_xy!(::Nothing, args...; kw...) = nothing
mask_immersed_field_xy!(field, value=zero(eltype(field.grid)); k, mask = peripheral_node) =
    mask_immersed_field_xy!(field, field.grid, location(field), value; k, mask)

mask_immersed_field_xy!(::Number, args...) = nothing

function mask_immersed_field_xy!(bop::BinaryOperation, value=zero(eltype(bop.b.grid)))
    mask_immersed_field_xy!(bop.a, value)
    mask_immersed_field_xy!(bop.b, value)
    return nothing
end

function mask_immersed_field_xy!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(*)}, value=zero(eltype(bop.b.grid)))
    mask_immersed_field_xy!(bop.a, value)
    isa(bop.a, Number) && mask_immersed_field_xy!(bop.b, value)
    return nothing
end

function mask_immersed_field_xy!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(/)}, value=zero(eltype(bop.b.grid)))
    mask_immersed_field_xy!(bop.a, value)
    isa(bop.a, Number) && mask_immersed_field_xy!(bop.b, Inf)
    return nothing
end

function mask_immersed_field_xy!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(^)}, value=zero(eltype(bop.b.grid)))
    mask_immersed_field_xy!(bop.a, value)
    mask_immersed_field_xy!(bop.b, one(bop.b.grid)) # if `a` is a number then the resulting field is not going to be zero
    return nothing
end

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

