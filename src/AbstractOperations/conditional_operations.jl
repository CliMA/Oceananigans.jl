using Oceananigans.Fields: OneField
import Oceananigans.Fields: condition_operand, conditional_length, set!

# For conditional reductions such as mean(u * v, condition = u .> 0))

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

@inline condition_operand(operand::AbstractField, condition, mask)    = condition_operand(location(operand)..., operand, operand.grid, condition, mask)
@inline condition_operand(LX, LY, LZ, operand, grid, condition, mask) = ConditionalOperation{LX, LY, LZ}(operand, grid, condition, mask)

@inline condition_onefield(c::ConditionalOperation{LX, LY, LZ}, mask) where {LX, LY, LZ} =
                              ConditionalOperation{LX, LY, LZ}(OneField(), c.grid, c.condition, mask)

# If we reduce a ConditionalOperation, we keep the same condition!
@inline condition_operand(operand::ConditionalOperation, ::Nothing, mask) = condition_operand(location(operand)..., operand, operand.grid, operand.condition, mask)
@inline condition_operand(operand::ConditionalOperation, condition, mask) = condition_operand(location(operand)..., operand, operand.grid, condition, mask)

@inline conditional_length(c::ConditionalOperation)       = sum(condition_onefield(c, 0))
@inline conditional_length(c::ConditionalOperation, dims) = sum(condition_onefield(c, 0); dims = dims)

Adapt.adapt_structure(to, c::ConditionalOperation{LX, LY, LZ}) where {LX, LY, LZ} =
            ConditionalOperation{LX, LY, LZ}(adapt(to, c.operand), 
                                     adapt(to, c.grid),
                                     adapt(to, c.condition),
                                     adapt(to, c.mask))

@inline function Base.getindex(c::ConditionalOperation, i, j, k) 
    return ifelse(get_condition(c.condition, i, j, k, c.grid, c), 
                  getindex(c.operand, i, j, k),
                  c.mask)
end

@inline concretize_condition!(c::ConditionalOperation) = set!(c.operand, c)

function concretize_condition(c::ConditionalOperation)
    f = similar(c.operand)
    set!(f, c)
    return f
end

@inline get_condition(condition, i, j, k, grid, args...)                = condition(i, j, k, grid, args...)
@inline get_condition(condition::AbstractArray, i, j, k, grid, args...) = condition[i, j, k]

