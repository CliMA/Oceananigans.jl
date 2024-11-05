using KernelAbstractions: @kernel, @index
using Statistics
using Oceananigans.AbstractOperations: BinaryOperation
using Oceananigans.Fields: location, XReducedField, YReducedField, ZReducedField, Field, ReducedField

instantiate(T::Type) = T()
instantiate(t) = t

mask_immersed_field!(field, grid, loc, value) = nothing
mask_immersed_field!(field::Field, value=zero(eltype(field.grid))) =
    mask_immersed_field!(field, field.grid, location(field), value)

mask_immersed_field!(::Number, args...) = nothing

function mask_immersed_field!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(+)}, value=zero(eltype(bop)))
    a_value = ifelse(bop.b isa Number, -bop.b, value)
    mask_immersed_field!(bop.a, a_value)

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

function mask_immersed_field_xy!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(+)}, value=zero(eltype(bop)))
    a_value = ifelse(bop.b isa Number, -bop.b, value)
    mask_immersed_field_xy!(bop.a, a_value)

    b_value = ifelse(bop.a isa Number, -bop.a, value)
    mask_immersed_field_xy!(bop.b, b_value)
    return nothing
end

function mask_immersed_field_xy!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(-)}, value=zero(eltype(bop)))
    a_value = ifelse(bop.b isa Number, bop.b, value)
    mask_immersed_field_xy!(bop.a, a_value)

    b_value = ifelse(bop.a isa Number, bop.a, value)
    mask_immersed_field_xy!(bop.b, b_value)
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
##### Masking a `ReducedField`
#####

# We mask a `ReducedField` if the entire reduced direction is immersed.
# This requires a sweep over the reduced direction

function mask_immersed_field!(field::ReducedField, grid, loc, value)
    loc = instantiate.(loc)
    launch!(architecture(field), grid, size(field), _mask_immersed_reduced_field!, field, loc, grid, value)
    return nothing
end

@kernel function _mask_immersed_reduced_field!(field, loc, grid, value)
    i, j, k = @index(Global, NTuple)

    mask = mask_reduced_x_direction!(field, i, j, k, grid, loc, true)
    mask = mask_reduced_y_direction!(field, i, j, k, grid, loc, mask)
    mask = mask_reduced_z_direction!(field, i, j, k, grid, loc, mask)

    @inbounds field[i, j, k] = ifelse(mask, value, field[i, j, k]) 
end

@inline mask_reduced_x_direction!(field, i, j, k, grid, loc, mask = true) = mask & peripheral_node(i, j, k, grid, loc...)
@inline mask_reduced_y_direction!(field, i, j, k, grid, loc, mask = true) = mask & peripheral_node(i, j, k, grid, loc...)
@inline mask_reduced_z_direction!(field, i, j, k, grid, loc, mask = true) = mask & peripheral_node(i, j, k, grid, loc...)

@inline function mask_reduced_x_direction!(::XReducedField, i₀, j, k, grid, loc, mask = true)
    for i in 1:size(grid, 1)
        mask = mask & peripheral_node(i, j, k, grid, loc...)
    end
    return mask
end

@inline function mask_reduced_y_direction!(::YReducedField, i, j₀, k, grid, loc, mask = true)
    for j in 1:size(grid, 2)
        mask = mask & peripheral_node(i, j, k, grid, loc...)
    end
    return mask
end

@inline function mask_reduced_z_direction!(::ZReducedField, i, j, k₀, grid, loc, mask = true)
    for k in 1:size(grid, 3)
        mask = mask & peripheral_node(i, j, k, grid, loc...)
    end
    return mask
end

###
### Efficient masking for `OnlyZReducedField` and an `AbstractGridFittedBoundary`
###

const AGFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}

const CenterOrFace = Union{Center, Face}
const OnlyZReducedField = Field{<:CenterOrFace, <:CenterOrFace, Nothing}

# Does not require a sweep
mask_immersed_field!(field::OnlyZReducedField, grid::AGFBIBG, loc, value) = 
    mask_immersed_field_xy!(field, grid, loc, value; k=size(grid, 3), mask=peripheral_node)
    
#####
##### Masking for GridFittedBoundary
#####

@inline scalar_mask(i, j, k, grid, ::AbstractGridFittedBoundary, LX, LY, LZ, value, field, mask=peripheral_node) =
    @inbounds ifelse(mask(i, j, k, grid, LX, LY, LZ), value, field[i, j, k])

