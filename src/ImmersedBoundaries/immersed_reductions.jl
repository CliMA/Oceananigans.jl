using Oceananigans.Fields: AbstractField, OneField, indices
using Oceananigans.AbstractOperations: KernelFunctionOperation

import Oceananigans.AbstractOperations: ConditionalOperation, evaluate_condition, validate_condition
import Oceananigans.Fields: condition_operand, conditional_length

# ImmersedReducedFields
const XIRF = AbstractField{Nothing, <:Any, <:Any, <:ImmersedBoundaryGrid}
const YIRF = AbstractField{<:Any, Nothing, <:Any, <:ImmersedBoundaryGrid}
const ZIRF = AbstractField{<:Any, <:Any, Nothing, <:ImmersedBoundaryGrid}

const YZIRF = AbstractField{<:Any, Nothing, Nothing, <:ImmersedBoundaryGrid}
const XZIRF = AbstractField{Nothing, <:Any, Nothing, <:ImmersedBoundaryGrid}
const XYIRF = AbstractField{Nothing, Nothing, <:Any, <:ImmersedBoundaryGrid}

const XYZIRF = AbstractField{Nothing, Nothing, Nothing, <:ImmersedBoundaryGrid}

const IRF = Union{XIRF, YIRF, ZIRF, YZIRF, XZIRF, XYIRF, XYZIRF}

#####
##### Reduction operations involving immersed boundary grids exclude the immersed periphery,
##### which includes both external nodes and nodes on the immersed interface.
#####

struct NotImmersed{F} <: Function
    condition :: F
end

NotImmersed() = NotImmersed(nothing)
Base.summary(::NotImmersed{Nothing}) = "NotImmersed()"
Base.summary(ni::NotImmersed) = string("NotImmersed(", summary(ni.condition), ")")
Base.size(ni::NotImmersed{<:AbstractArray}) = size(ni.condition)

validate_condition(cond::NotImmersed{<:AbstractArray}, ::OneField) = cond

validate_condition(cond::NotImmersed{<:AbstractArray}, operand::AbstractField) =
    validate_condition(cond.condition, operand)

"Adapt `NotImmersed` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, ni::NotImmersed) = NotImmersed(Adapt.adapt(to, ni.condition))

# ImmersedField
const IF = AbstractField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}

function ConditionalOperation(operand::IF;
                              func = nothing,
                              condition = nothing,
                              mask = zero(eltype(operand)))

    condition = validate_condition(condition, operand)

    if condition isa NotImmersed || condition isa NotImmersedColumn
        immersed_condition = condition # it's immersed enough
    elseif operand isa IRF
        immersed_condition = NotImmersedColumn(immersed_column(operand), condition)
    else
        immersed_condition = NotImmersed(condition)
    end
    LX, LY, LZ = location(operand)
    grid = operand.grid

    return ConditionalOperation{LX, LY, LZ}(operand, func, grid, immersed_condition, mask)
end

@inline conditional_length(c::IF) = conditional_length(condition_operand(identity, c, NotImmersed(), 0))
@inline conditional_length(c::IF, dims) = conditional_length(condition_operand(identity, c, NotImmersed(), 0), dims)

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

struct NotImmersedColumn{F, IC} <: Function
    immersed_column :: IC
    condition :: F
end

Base.summary(nic::NotImmersedColumn) = string("NotImmersedColumn(",
                                              summary(nic.immersed_column), ", ",
                                              summary(nic.condition), ") ")

Base.show(io::IO, nic::NotImmersedColumn) = print(io, Base.summary(nic))

NotImmersedColumn(immersed_column) = NotImmersedColumn(immersed_column, nothing)

"Adapt `NotImmersed` to work on the GPU via CUDAnative and CUDAdrv."
function Adapt.adapt_structure(to, nic::NotImmersedColumn)
    return NotImmersedColumn(Adapt.adapt(to, nic.immersed_column),
                             Adapt.adapt(to, nic.condition))
end

using Oceananigans.Fields: reduced_dimensions, OneField
using Oceananigans.AbstractOperations: ConditionalOperation

@inline function condition_operand(func, op::IRF, condition, mask)
    immersed_condition = NotImmersedColumn(immersed_column(op), condition)
    return ConditionalOperation(op; func, condition=immersed_condition, mask)
end

@inline function condition_operand(::Nothing, op::IRF, ::Nothing, mask)
    immersed_condition = NotImmersedColumn(immersed_column(op), nothing)
    return ConditionalOperation(op; func=nothing, condition=immersed_condition, mask)
end

@inline function condition_operand(func, op::IF, condition, mask)
    immersed_condition = NotImmersed(condition)
    return ConditionalOperation(op; func, condition=immersed_condition, mask)
end

@inline function condition_operand(::Nothing, op::IF, ::Nothing, mask)
    immersed_condition = NotImmersed()
    return ConditionalOperation(op; func=nothing, condition=immersed_condition, mask)
end

@inline function condition_operand(func, operand::IF, condition::AbstractArray, mask)
    condition = on_architecture(architecture(operand.grid), condition)
    immersed_condition = NotImmersed(condition)
    return ConditionalOperation(operand; func, condition=immersed_condition, mask)
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

@inline function evaluate_condition(nic::NotImmersedColumn, i, j, k,
                                    grid::ImmersedBoundaryGrid,
                                    ::ConditionalOperation, args...)
    immersed = is_immersed_column(i, j, k, nic.immersed_column)
    value = !immersed & evaluate_condition(nic.condition, i, j, k, grid, args...)
    return value
end

@inline is_immersed_column(i, j, k, column) = @inbounds column[i, j, k] == 0

const NICO{LX, LY, LZ, F, C} = Union{
    ConditionalOperation{LX, LY, LZ, F, C, <:NotImmersed, <:ImmersedBoundaryGrid},
    ConditionalOperation{LX, LY, LZ, F, C, <:NotImmersedColumn, <:ImmersedBoundaryGrid},
}
@inline conditional_length(c::NICO) = sum(conditional_one(c, 0))
@inline conditional_length(c::NICO, dims) = sum(conditional_one(c, 0); dims = dims)
