import Oceananigans.Fields: mask_operator

# For conditional reductions such as mean(mask_operator(u, 0, u .> 0))

struct MaskedOperation{LX, LY, LZ, O, G, M, C, T} <: AbstractOperation{LX, LY, LZ, G, T} 
    operand  :: O
    grid :: G
    mask :: M
    condition :: C
 
     function MaskedOperation{LX, LY, LZ}(operand::O, grid::G, mask::M, condition::C) where {LX, LY, LZ, O, G, M, C}
         T = eltype(operand)
         return new{LX, LY, LZ, O, G, M, C, T}(operand, grid, mask, condition)
     end
end
 
@inline mask_operator(operand, mask, ::Nothing) = operand 
@inline mask_operator(operand::AbstractField, mask, condition)    = mask_operator(location(operand)..., operand.grid, mask, condition)
@inline mask_operator(LX, LY, LZ, operand, grid, mask, condition) = MaskedOperation{LX, LY, LZ}(operand, grid, mask, condition)

Adapt.adapt_structure(to, mo::MaskedOperation{LX, LY, LZ}) where {LX, LY, LZ} =
            MaskedOperation{LX, LY, LZ}(adapt(to, mo.operand), 
                                     adapt(to, mo.grid),
                                     adapt(to, mo.mask),
                                     adapt(to, mo.condition))

@inline Base.size(mo::MaskedOperation) = Base.size(mo.operand)

@inline function Base.getindex(mo::MaskedOperation{LX, LY, LZ}, i, j, k) where {LX, LY, LZ}
    return ifelse(condition(i, j, k, mo), 
                  get_mask(mo.mask, i, j, k), 
                  getindex(mo.operand, i, j, k))
end

@inline get_mask(mask::Number       , i, j, k) = mask
@inline get_mask(mask::AbstractArray, i, j, k) = mask[i, j, k]
@inline get_mask(mask::Base.Callable, i, j, k) = mask(i, j, k)
