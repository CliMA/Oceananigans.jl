using Oceananigans.Fields: AbstractField

import Oceananigans.AbstractOperations: ConditionalOperation, get_condition, truefunc
import Oceananigans.Fields: condition_operand, conditional_length

# ###
# ###  reduction operations involving immersed boundary grids exclude the immersed region 
# ### (we exclude also the values on the faces of the immersed boundary with `solid_interface`)
# ###

struct NotImmersed{F} <: Function
    func :: F
end

# ImmersedField
const IF = AbstractField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}

@inline condition_operand(func::Function,         op::IF, cond,      mask) = ConditionalOperation(op; func, condition=NotImmersed(cond),     mask)
@inline condition_operand(func::Function,         op::IF, ::Nothing, mask) = ConditionalOperation(op; func, condition=NotImmersed(truefunc), mask)
@inline condition_operand(func::typeof(identity), op::IF, ::Nothing, mask) = ConditionalOperation(op; func, condition=NotImmersed(truefunc), mask)

@inline conditional_length(c::IF)       = conditional_length(condition_operand(identity, c, nothing, 0))
@inline conditional_length(c::IF, dims) = conditional_length(condition_operand(identity, c, nothing, 0), dims)

@inline function get_condition(condition::NotImmersed, i, j, k, ibg, co::ConditionalOperation, args...)
    LX, LY, LZ = location(co)
    return get_condition(condition.func, i, j, k, ibg, args...) & !(solid_interface(LX(), LY(), LZ(), i, j, k, ibg))
end 

