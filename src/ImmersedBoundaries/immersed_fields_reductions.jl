using Oceananigans.Fields: AbstractField, ReducedField
using Oceananigans.AbstractOperations: AbstractOperation, ConditionalOperation
using CUDA: @allowscalar

import Oceananigans.AbstractOperations: get_condition
import Oceananigans.Fields: condition_operand, conditional_length

# ###
# ###  reduction operations involving immersed boundary grids exclude the immersed region 
# ### (we exclude also the values on the faces of the immersed boundary with `solid_interface`)
# ###

const ImmersedField = AbstractField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}

@inline function get_condition(condition, i, j, k, 
                               ibg :: ImmersedBoundaryGrid, 
                               co :: ConditionalOperation{LX, LY, LZ}) where {LX, LY, LZ}
    return get_condition(condition, i, j, k, ibg.grid, co) & !solid_interface(LX(), LY(), LZ(), i, j, k, ibg)
end 

@inline condition_operand(c::ImmersedField, ::Nothing, mask) = condition_operand(location(c)..., c, c.grid, mask, neutral_func)
@inline neutral_func(i, j, k, grid) = true

@inline conditional_length(c::ImmersedField) = conditional_length(condition_operand(c, nothing, 0))







# Statistics.dot(a::ImmersedField, b::Field) = Statistics.dot(condition_operand(a, 0), b)
# Statistics.dot(a::Field, b::ImmersedField) = Statistics.dot(condition_operand(a, 0), b)

# function Statistics.norm(c::ImmersedField)
#     r = zeros(c.grid, 1)
#     Base.mapreducedim!(x -> x * x, +, r, condition_operator(c, nothing, 0))
#     return @allowscalar sqrt(r[1])
# end
