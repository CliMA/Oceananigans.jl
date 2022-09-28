using Oceananigans.Fields: AbstractField, offset_compute_index, indices

import Oceananigans.AbstractOperations: ConditionalOperation, get_condition, truefunc
import Oceananigans.Fields: condition_operand, conditional_length

#####
##### Reduction operations involving immersed boundary grids exclude the immersed periphery,
##### which includes both external nodes and nodes on the immersed interface.
#####

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
    return get_condition(condition.func, i, j, k, ibg, args...) & !(immersed_peripheral_node(i, j, k, ibg, LX(), LY(), LZ()))
end 

#####
##### Reduction operations on Reduced Fields have to test if entirety of the immersed direction is immersed to exclude it
#####

# struct NotImmersedReduced{F, D} <: Function
#     func :: F
#     immersed_dimensions :: D
# end

# function NotImmersedReduced(func; location = ())

# # ImmersedReducedFields
# const IRF = ReducedField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}

# @inline condition_operand(func::Function,         op::IRF, cond,      mask) = ConditionalOperation(op; func, condition=NotImmersedReduced(cond),     mask)
# @inline condition_operand(func::Function,         op::IRF, ::Nothing, mask) = ConditionalOperation(op; func, condition=NotImmersedReduced(truefunc), mask)
# @inline condition_operand(func::typeof(identity), op::IRF, ::Nothing, mask) = ConditionalOperation(op; func, condition=NotImmersedReduced(truefunc), mask)

# @inline function get_condition(condition::NotImmersedReduced, i, j, k, ibg, co::ConditionalOperation, args...)
#     LX, LY, LZ = location(co)
#     return get_condition(condition.func, i, j, k, ibg, args...) & !immersed_column(i, j, k, ibg, LX(), LY(), LZ(), condition))
# end 

# @inline function immersed_column(i, j, k, ibg, reduced_dims, )