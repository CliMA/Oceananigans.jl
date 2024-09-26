using Oceananigans.Fields: AbstractField, indices

import Oceananigans.AbstractOperations: ConditionalOperation, evaluate_condition
import Oceananigans.Fields: condition_operand, conditional_length

#####
##### Reduction operations involving immersed boundary grids exclude the immersed periphery,
##### which includes both external nodes and nodes on the immersed interface.
#####

@inline truefunc(args...) = true

struct NotImmersed{F} <: Function
    condition :: F
end

NotImmersed() = NotImmersed(nothing)
Base.summary(::NotImmersed{Nothing}) = "NotImmersed()"
Base.summary(::NotImmersed) = string("NotImmersed(", summary(condition), ")")

# ImmersedField
const IF = AbstractField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}

function ConditionalOperation(operand::IF;
                              func = nothing,
                              condition = nothing,
                              mask = zero(eltype(operand)))

    immersed_condition = NotImmersed(condition)
    LX, LY, LZ = location(operand)
    return ConditionalOperation{LX, LY, LZ}(operand, func, operand.grid, immersed_condition, mask)
end

@inline conditional_length(c::IF) = conditional_length(condition_operand(c, nothing, 0))
@inline conditional_length(c::IF, dims) = conditional_length(condition_operand(c, nothing, 0), dims)

@inline function evaluate_condition(::NotImmersed{Nothing}, i, j, k,
                                    grid::ImmersedBoundaryGrid,
                                    co::ConditionalOperation, args...)

    ℓx, ℓy, ℓz = map(instantiate, location(co))
    return peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz)
end 

@inline function evaluate_condition(condition::NotImmersed, i, j, k,
                                    grid::ImmersedBoundaryGrid,
                                    co::ConditionalOperation, args...)

    ℓx, ℓy, ℓz = map(instantiate, location(co))
    immersed = peripheral_node(i, j, k, ibg, ℓx, ℓy, ℓz)
    return !immersed & evaluate_condition(ni.condition, i, j, k, ibg, args...)
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
    grid         = field.grid
    reduced_dims = reduced_dimensions(field)
    LX, LY, LZ   = map(center_to_nothing, location(field))
    one_field    = ConditionalOperation{LX, LY, LZ}(OneField(Int), identity, grid, NotImmersed(truefunc), zero(grid))
    return sum(one_field, dims=reduced_dims)
end

@inline center_to_nothing(::Type{Face})    = Face
@inline center_to_nothing(::Type{Center})  = Center
@inline center_to_nothing(::Type{Nothing}) = Center

@inline function evaluate_condition(nic::NotImmersedColumn, i, j, k, ibg, co::ConditionalOperation, args...)
    LX, LY, LZ = location(co)
    return evaluate_condition(ni.condition, i, j, k, ibg, args...) & !(is_immersed_column(i, j, k, nic.immersed_column))
end 

@inline is_immersed_column(i, j, k, column) = @inbounds column[i, j, k] == 0
