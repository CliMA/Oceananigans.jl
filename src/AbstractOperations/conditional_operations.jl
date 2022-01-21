import Oceananigans.Fields: condition_operand, conditional_length

# For conditional reductions such as mean(condition_operand(u * v, 0, u > 0))

struct ConditionalOperation{LX, LY, LZ, O, G, C, M, T} <: AbstractOperation{LX, LY, LZ, G, T} 
    operand :: O
    grid :: G
    condition :: C
    mask :: M
 
     function ConditionalOperation{LX, LY, LZ}(operand::O, grid::G, condition::C, mask::M) where {LX, LY, LZ, O, G, C, M}
         T = eltype(operand)
         return new{LX, LY, LZ, O, G, C, M, T}(operand, grid, condition, mask)
     end
end
 
@inline condition_operand(operand::AbstractField, condition, mask)    = condition_operand(location(operand)..., operand.grid, condition, mask)
@inline condition_operand(LX, LY, LZ, operand, grid, condition, mask) = ConditionalOperation{LX, LY, LZ}(operand, grid, condition, mask)

@inline conditional_length(c::ConditionalOperation)       = sum(c / c)
@inline conditional_length(c::ConditionalOperation, dims) = sum(f, c / c; dims = dims)

Adapt.adapt_structure(to, mo::ConditionalOperation{LX, LY, LZ}) where {LX, LY, LZ} =
            ConditionalOperation{LX, LY, LZ}(adapt(to, mo.operand), 
                                     adapt(to, mo.grid),
                                     adapt(to, mo.mask),
                                     adapt(to, mo.condition))

@inline function Base.getindex(mo::ConditionalOperation, i, j, k) 
    return ifelse(get_condition(mo.condition, i, j, k, mo.grid, mo), 
                  getindex(mo.operand, i, j, k),
                  mo.mask)
end

@inline get_condition(condition, i, j, k, grid, args...)                = condition(i, j, k, grid, args...)
@inline get_condition(condition::AbstractArray, i, j, k, grid, args...) = condition[i, j, k]