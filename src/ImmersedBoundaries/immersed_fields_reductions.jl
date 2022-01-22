using Oceananigans.Fields: AbstractField
using Oceananigans.AbstractOperations: AbstractOperation, ConditionalOperation

import Oceananigans.AbstractOperations: get_condition
import Oceananigans.Fields: condition_operand, conditional_length

# ###
# ###  reduction operations involving immersed boundary grids exclude the immersed region 
# ### (we exclude also the values on the faces of the immersed boundary with `solid_interface`)
# ###

const ImmersedField = AbstractField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}

struct NotImmersed{F} <: Function
    func :: F
end

@inline condition_operand(c::ImmersedField, ::Nothing, mask) = condition_operand(location(c)..., c, c.grid, NotImmersed(neutral_func), mask)
@inline condition_operand(c::ImmersedField, condition, mask) = condition_operand(location(c)..., c, c.grid, NotImmersed(condition), mask)

@inline neutral_func(args...) = true

@inline conditional_length(c::ImmersedField)       = conditional_length(condition_operand(c, nothing, 0))
@inline conditional_length(c::ImmersedField, dims) = conditional_length(condition_operand(c, nothing, 0), dims)

@inline function get_condition(condition::NotImmersed, i, j, k, 
                               ibg, co::ConditionalOperation{LX, LY, LZ}, args...) where {LX, LY, LZ}
    return get_condition(condition.func, i, j, k, ibg, args...) && !(solid_interface(LX(), LY(), LZ(), i, j, k, ibg))
end 

Statistics.dot(a::ImmersedField, b::Field) = Statistics.dot(condition_operand(a, nothing, 0), b)
Statistics.dot(a::Field, b::ImmersedField) = Statistics.dot(a, condition_operand(b, nothing, 0))

Statistics.dot(a::ImmersedField, b::ImmersedField) = Statistics.dot(condition_operand(a, nothing, 0),
                                                                    condition_operand(b, nothing, 0))
