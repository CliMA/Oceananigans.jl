using Oceananigans.Fields: AbstractField

import Oceananigans.AbstractOperations: ConditionalOperation, get_condition, truefunc
import Oceananigans.Fields: condition_operand, conditional_length

# ###
# ###  reduction operations involving immersed boundary grids exclude the immersed region 
# ### (we exclude also the values on the faces of the immersed boundary with `solid_interface`)
# ###

const ImmersedField = AbstractField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}

struct NotImmersed{F} <: Function
    func :: F
end

@inline condition_operand(func::Function, operand::ImmersedField, condition, mask) = ConditionalOperation(operand; func, condition = NotImmersed(condition), mask)
@inline condition_operand(func::Function, operand::ImmersedField, ::Nothing, mask) = ConditionalOperation(operand; func, condition = NotImmersed(truefunc), mask)

@inline condition_operand(func::typeof(identity), operand::ImmersedField, condition, mask) = ConditionalOperation(operand; func, condition = NotImmersed(condition), mask)
@inline condition_operand(func::typeof(identity), operand::ImmersedField, ::Nothing, mask) = ConditionalOperation(operand; func, condition = NotImmersed(truefunc), mask)

@inline conditional_length(c::ImmersedField)       = conditional_length(condition_operand(identity, c, nothing, 0))
@inline conditional_length(c::ImmersedField, dims) = conditional_length(condition_operand(identity, c, nothing, 0), dims)

@inline function get_condition(condition::NotImmersed, i, j, k, ibg, co::ConditionalOperation, args...)
    LX, LY, LZ = location(condition)
    return get_condition(condition.func, i, j, k, ibg, args...) & !(solid_interface(LX(), LY(), LZ(), i, j, k, ibg))
end 
