using Oceananigans.Fields: AbstractField, indices

import Oceananigans.AbstractOperations: ConditionalOperation, evaluate_condition, validate_condition
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
Base.size(ni::NotImmersed{<:AbstractArray}) = size(ni.condition)

validate_condition(cond::NotImmersed{<:AbstractArray}, operand::AbstractField) = validate_condition(cond.conditio, operand)


# ImmersedField
const IF = AbstractField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}

function ConditionalOperation(operand::IF;
                              func = nothing,
                              condition = nothing,
                              mask = zero(eltype(operand)))

    immersed_condition = NotImmersed(condition)
    LX, LY, LZ = location(operand)
    grid = operand.grid
    return ConditionalOperation{LX, LY, LZ}(operand, func, grid, immersed_condition, mask)
end

@inline conditional_length(c::IF) = conditional_length(condition_operand(c, nothing, 0))
@inline conditional_length(c::IF, dims) = conditional_length(condition_operand(c, nothing, 0), dims)

@inline function evaluate_condition(::NotImmersed{Nothing},
                                    i, j, k,
                                    grid::ImmersedBoundaryGrid,
                                    co::ConditionalOperation) #, args...)

    ℓx, ℓy, ℓz = map(instantiate, location(co))
    immersed = immersed_peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz) | inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)
    return !immersed
end

@inline function evaluate_condition(ni::NotImmersed,
                                    i, j, k,
                                    grid::ImmersedBoundaryGrid,
                                    co::ConditionalOperation, args...)

    ℓx, ℓy, ℓz = map(instantiate, location(co))
    immersed = immersed_peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz) | inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)
    return !immersed & evaluate_condition(ni.condition, i, j, k, grid, co, args...)
end

@inline function evaluate_condition(condition::NotImmersed, i::AbstractArray, j::AbstractArray, k::AbstractArray, ibg, co::ConditionalOperation, args...)
    ℓx, ℓy, ℓz = map(instantiate, location(co))
    immersed = immersed_peripheral_node(i, j, k, ibg, ℓx, ℓy, ℓz) .| inactive_node(i, j, k, ibg, ℓx, ℓy, ℓz)
    return Base.broadcast(!, immersed) .& evaluate_condition(condition.func, i, j, k, ibg, args...)
end

#####
##### Reduction operations on Reduced Fields test the immersed condition on the entirety of the immersed direction
#####

struct NotImmersedColumn{F, IC} <:Function
    immersed_column :: IC
    condition :: F
end

NotImmersedColumn(immersed_column) = NotImmersedColumn(immersed_column, nothing)

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

@inline function condition_operand(func, op::IRF, condition, mask)
    immersed_condition = NotImmersedColumn(immersed_column(op), condition)
    return ConditionalOperation(op; func, condition, mask)
end

@inline function immersed_column(field::IRF)
    grid         = field.grid
    reduced_dims = reduced_dimensions(field)
    LX, LY, LZ   = map(center_to_nothing, location(field))
    one_field    = ConditionalOperation{LX, LY, LZ}(OneField(Int), identity, grid, NotImmersed(), zero(grid))
    return sum(one_field, dims=reduced_dims)
end

@inline center_to_nothing(::Type{Face})    = Face
@inline center_to_nothing(::Type{Center})  = Center
@inline center_to_nothing(::Type{Nothing}) = Center

@inline function evaluate_condition(nic::NotImmersedColumn,
                                    i, j, k,
                                    grid::ImmersedBoundaryGrid,
                                    co::ConditionalOperation, args...)
    LX, LY, LZ = location(co)
    immersed = is_immersed_column(i, j, k, nic.immersed_column)
    return !immersed & evaluate_condition(nic.condition, i, j, k, grid, args...)
end

@inline is_immersed_column(i, j, k, column) = @inbounds column[i, j, k] == 0

