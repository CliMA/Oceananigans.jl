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
##### Reduction operations on Reduced Fields test the immersed condition on the entirety of the immersed direction
#####

struct NotImmersedColumn{IC, F} <:Function
    immersed_column :: IC
    func :: F
end

using Oceananigans.Fields: reduced_dimensions, OneField
using Oceananigans.AbstractOperations: ConditionalOperation

# ImmersedReducedFields
const XIRF = AbstractField{Nothing, <:Any, <:Any, <:ImmersedBoundaryGrid}
const YIRF = AbstractField{<:Any, Nothing, <:Any, <:ImmersedBoundaryGrid}
const ZIRF = AbstractField{<:Any, <:Any, Nothing, <:ImmersedBoundaryGrid}

const YZIRF = AbstractField{<:Any, Nothing, Nothing, <:ImmersedBoundaryGrid}
const XZIRF = AbstractField{Nothing, <:Any, Nothing, <:ImmersedBoundaryGrid}
const XYIRF = AbstractField{Nothing, Nothing, <:Any, <:ImmersedBoundaryGrid}

const XYZIRF = AbstractField{Nothing, Nothing, Nothing, <:ImmersedBoundaryGrid}

const IRF = Union{XIRF, YIRF, ZIRF, YZIRF, XZIRF, XYIRF, XYZIRF}

@inline condition_operand(func::Function,         op::IRF, cond,      mask) = ConditionalOperation(op; func, condition=NotImmersedColumn(immersed_column(op), cond    ), mask)
@inline condition_operand(func::Function,         op::IRF, ::Nothing, mask) = ConditionalOperation(op; func, condition=NotImmersedColumn(immersed_column(op), truefunc), mask)
@inline condition_operand(func::typeof(identity), op::IRF, ::Nothing, mask) = ConditionalOperation(op; func, condition=NotImmersedColumn(immersed_column(op), truefunc), mask)

@inline function immersed_column(field::IRF)
    reduced_dims = reduced_dimensions(field)
    one_field    = ConditionalOperation{location(field)...}(OneField(Int), identity, field.grid, NotImmersed(truefunc), 0.0)

    return sum(one_field, dims = reduced_dims)
end

@inline function get_condition(condition::NotImmersedColumn, i, j, k, ibg, co::ConditionalOperation, args...)
    LX, LY, LZ = location(co)
    return get_condition(condition.func, i, j, k, ibg, args...) & !(is_immersed_column(i, j, k, condition.immersed_column))
end 

is_immersed_column(i, j, k, column) = column[i, j, k] == 0
