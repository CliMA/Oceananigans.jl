using Oceananigans.Fields: AbstractField

import Oceananigans.AbstractOperations: ConditionalOperation, get_condition
import Oceananigans.Fields: condition_operand, conditional_length

# ###
# ###  reduction operations involving immersed boundary grids exclude the immersed region 
# ### (we exclude also the values on the faces of the immersed boundary with `solid_interface`)
# ###

const ImmersedField = AbstractField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}

struct NotImmersed{F} <: Function
    func :: F
end

@inline condition_operand(c::ImmersedField, ::Nothing, mask) = condition_operand(c, NotImmersed(neutral_func), mask)
@inline condition_operand(c::ImmersedField, condition, mask) = condition_operand(c, NotImmersed(condition), mask)

@inline condition_operand(f::Function, c::ImmersedField, ::Nothing, mask) = condition_operand(f, c, NotImmersed(neutral_func), mask)
@inline condition_operand(f::Function, c::ImmersedField, condition, mask) = condition_operand(f, c, NotImmersed(condition), mask)

@inline neutral_func(args...) = true

@inline conditional_length(c::ImmersedField)       = conditional_length(condition_operand(c, nothing, 0))
@inline conditional_length(c::ImmersedField, dims) = conditional_length(condition_operand(c, nothing, 0), dims)

@inline function get_condition(condition::NotImmersed, i, j, k, 
                               ibg, co::ConditionalOperation{LX, LY, LZ}, args...) where {LX, LY, LZ}
    return get_condition(condition.func, i, j, k, ibg, args...) & !(solid_interface(LX(), LY(), LZ(), i, j, k, ibg))
end 