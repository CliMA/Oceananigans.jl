using KernelAbstractions: @kernel, @index
using Statistics
using Oceananigans.AbstractOperations: BinaryOperation
using Oceananigans.Fields: location, XReducedField, YReducedField, ZReducedField, Field, ReducedField
using Oceananigans.Fields: ConstantField, OneField, ZeroField

instantiate(T::Type) = T()
instantiate(t) = t

# No masking for constant fields, numbers or nothing
mask_immersed_field!(::OneField, args...; kw...) = nothing
mask_immersed_field!(::ZeroField, args...; kw...) = nothing
mask_immersed_field!(::ConstantField, args...; kw...) = nothing
mask_immersed_field!(::Number, args...; kw...) = nothing
mask_immersed_field!(::Nothing, args...; kw...) = nothing

# No masking for constant fields, numbers or nothing
mask_immersed_field_xy!(::OneField, args...; kw...) = nothing
mask_immersed_field_xy!(::ZeroField, args...; kw...) = nothing
mask_immersed_field_xy!(::ConstantField, args...; kw...) = nothing
mask_immersed_field_xy!(::Number, args...; kw...) = nothing
mask_immersed_field_xy!(::Nothing, args...; kw...) = nothing

mask_immersed_field!(field::Field, value=zero(eltype(field.grid)); mask=peripheral_node) =
    mask_immersed_field!(field, field.grid, location(field), value, mask)

function mask_immersed_field!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(+)}, value=zero(eltype(bop)); mask=peripheral_node)
    a_value = ifelse(bop.b isa Number, -bop.b, value)
    mask_immersed_field!(bop.a, a_value; mask)

    b_value = ifelse(bop.a isa Number, -bop.a, value)
    mask_immersed_field!(bop.b, b_value; mask)
    return nothing
end

function mask_immersed_field!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(-)}, value=zero(eltype(bop)); mask=peripheral_node)
    a_value = ifelse(bop.b isa Number, bop.b, value)
    mask_immersed_field!(bop.a, a_value; mask)

    b_value = ifelse(bop.a isa Number, bop.a, value)
    mask_immersed_field!(bop.b, b_value; mask)
    return nothing
end

# Fallback
mask_immersed_field!(field, grid, loc, value, mask) = nothing

"""
    mask_immersed_field!(field::Field, grid::ImmersedBoundaryGrid, loc, value)

masks `field` defined on `grid` with a value `val` at locations where `peripheral_node` evaluates to `true`
"""
function mask_immersed_field!(field::Field, grid::ImmersedBoundaryGrid, loc, value, mask)
    arch = architecture(field)
    loc  = instantiate.(loc)
    launch!(arch, grid, :xyz, _mask_immersed_field!, field, loc, grid, value, mask)
    return nothing
end

@kernel function _mask_immersed_field!(field, (ℓx, ℓy, ℓz), grid, value, mask)
    i, j, k = @index(Global, NTuple)
    masked  = mask(i, j, k, grid, ℓx, ℓy, ℓz)
    @inbounds field[i, j, k] = ifelse(masked, value, field[i, j, k])
end

mask_immersed_field_xy!(field, value=zero(eltype(field.grid)); k, mask=peripheral_node) =
    mask_immersed_field_xy!(field, field.grid, location(field), value, k, mask)

function mask_immersed_field_xy!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(+)}, value=zero(eltype(bop)); k, mask=peripheral_node)
    a_value = ifelse(bop.b isa Number, -bop.b, value)
    mask_immersed_field_xy!(bop.a, a_value; k, mask)

    b_value = ifelse(bop.a isa Number, -bop.a, value)
    mask_immersed_field_xy!(bop.b, b_value; k, mask)
    return nothing
end

function mask_immersed_field_xy!(bop::BinaryOperation{<:Any, <:Any, <:Any, typeof(-)}, value=zero(eltype(bop)); k, mask=peripheral_node)
    a_value = ifelse(bop.b isa Number, bop.b, value)
    mask_immersed_field_xy!(bop.a, a_value; k, mask)

    b_value = ifelse(bop.a isa Number, bop.a, value)
    mask_immersed_field_xy!(bop.b, b_value; k, mask)
    return nothing
end

# Fallback
mask_immersed_field_xy!(field, grid, loc, value, k, mask) = nothing

"""
    mask_immersed_field_xy!(field::Field, grid::ImmersedBoundaryGrid, loc, value; k, mask=peripheral_node)

Mask `field` on `grid` with a `value` on the slices `[:, :, k]` where `mask` is `true`.
"""
function mask_immersed_field_xy!(field::Field, grid::ImmersedBoundaryGrid, loc, value, k, mask)
    arch = architecture(field)
    loc  = instantiate.(loc)
    return launch!(arch, grid, :xy, _mask_immersed_field_xy!, field, loc, grid, value, k, mask)
end

@kernel function _mask_immersed_field_xy!(field, (ℓx, ℓy, ℓz), grid, value, k, mask)
    i, j = @index(Global, NTuple)
    masked = mask(i, j, k, grid, ℓx, ℓy, ℓz)
    @inbounds field[i, j, k] = ifelse(masked, value, field[i, j, k])
end

#####
##### Masking a `ReducedField`
#####

# We mask a `ReducedField` if the entire reduced direction is immersed.
# This requires a sweep over the reduced direction
function mask_immersed_field!(field::ReducedField, grid::ImmersedBoundaryGrid, loc, value, mask)
    loc  = instantiate.(loc)
    dims = reduced_dimensions(field)
    launch!(architecture(field), grid, size(field), _mask_immersed_reduced_field!, field, dims, loc, grid, value, mask)
    return nothing
end

@kernel function _mask_immersed_reduced_field!(field, dims, (ℓx, ℓy, ℓz), grid, value, mask)
    i₀, j₀, k₀ = @index(Global, NTuple)
    masked = true
    irange = inactive_search_range(i₀, grid, 1, dims)
    jrange = inactive_search_range(j₀, grid, 2, dims)
    krange = inactive_search_range(k₀, grid, 3, dims)
    
    # The loop activates over the whole direction only if reduced directions
    for i in irange, j in jrange, k in krange
        masked = masked & mask(i, j, k, grid, ℓx, ℓy, ℓz) 
    end

    @inbounds field[i₀, j₀, k₀] = ifelse(masked, value, field[i₀, j₀, k₀]) 
end

@inline inactive_search_range(i, grid, dim, dims) = ifelse(dim ∈ dims, 1:size(grid, dim), i:i)

###
### Efficient masking for `OnlyZReducedField` and an `AbstractGridFittedBottom`
###

const AGFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}

const CenterOrFace = Union{Center, Face}
const OnlyZReducedField = Field{<:CenterOrFace, <:CenterOrFace, Nothing}

# Does not require a sweep
mask_immersed_field!(field::OnlyZReducedField, grid::AGFBIBG, loc, value, mask) = 
    mask_immersed_field_xy!(field, grid, loc, value, size(grid, 3), mask)