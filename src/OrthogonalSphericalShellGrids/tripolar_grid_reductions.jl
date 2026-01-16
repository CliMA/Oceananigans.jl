using Oceananigans.AbstractOperations: ConditionalOperation
using Oceananigans.Fields: AbstractField, OneField, instantiated_location
using Oceananigans.ImmersedBoundaries: NotImmersed, NotImmersedColumn

import Oceananigans.AbstractOperations: evaluate_condition, validate_condition
import Oceananigans.Fields: condition_operand, conditional_length

#####
##### Reduction operations involving tripolar grids exclude the repeated row at the top 
##### of the domain for Fields located on `Center`s in meridional direction.
#####

struct ValidTripolarDomain{F} <: Function 
    condition :: F
end

ValidTripolarDomain() = ValidTripolarDomain(nothing)

Base.summary(::ValidTripolarDomain{Nothing}) = "ValidTripolarDomain()"
Base.summary(vd::ValidTripolarDomain) = string("ValidTripolarDomain(", summary(vd.condition), ")")
Base.size(vd::ValidTripolarDomain{<:AbstractArray}) = size(vd.condition)

validate_condition(cond::ValidTripolarDomain{<:AbstractArray}, ::OneField) = cond
validate_condition(cond::ValidTripolarDomain{<:AbstractArray}, operand::AbstractField) = validate_condition(cond.condition, operand)

"Adapt `ValidTripolarDomain` to work on the GPU via KernelAbstractions."
Adapt.adapt_structure(to, vd::ValidTripolarDomain) = ValidTripolarDomain(Adapt.adapt(to, vd.condition))

@inline function evaluate_condition(::ValidTripolarDomain{Nothing},
                                    i, j, k,
                                    grid::TripolarGridOfSomeKind,
                                    co::ConditionalOperation) #, args...)
    ℓx, ℓy, ℓz = Oceananigans.Fields.location(co)
    Nx, Ny, _ = size(grid)
    valid_domain = !((i > Nx÷2) & (j == Ny))
    return ifelse(ℓy == Face, true, valid_domain)
end

@inline function evaluate_condition(vd::ValidTripolarDomain,
                                    i, j, k,
                                    grid::TripolarGridOfSomeKind,
                                    co::ConditionalOperation, args...)

    valid_domain = evaluate_condition(ValidTripolarDomain(), i, j, k, grid, co)                                
    return valid_domain & evaluate_condition(vd.condition, i, j, k, grid, co, args...)
end

#####
##### conditional_operand extension
#####

ITG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}

const TF = Union{<:AbstractField{<:Any, <:Any, <:Any, <:TripolarGridOfSomeKind},
                 <:AbstractField{<:Any, <:Any, <:Any, <:ITG}}

@inline conditional_length(c::TF) = conditional_length(condition_operand(identity, c, ValidTripolarDomain(), 0))
@inline conditional_length(c::TF, dims) = conditional_length(condition_operand(identity, c, ValidTripolarDomain(), 0), dims)

condition_operand(::typeof(identity), op::TF, ::Nothing, mask) =
    condition_operand(nothing, op, nothing, mask)

@inline function condition_operand(::Nothing, op::TF, ::Nothing, mask)
    arch = architecture(op)

    if !(arch isa Distributed) || (arch.ranks[2] == arch.local_index[2]) # The last core
        tripolar_condition = ValidTripolarDomain()
    else # intermediate cores
        tripolar_condition = nothing
    end

    return ConditionalOperation(op; func=nothing, condition=tripolar_condition, mask)
end

@inline function condition_operand(func, op::TF, condition, mask)
    arch = architecture(op)

    if !(arch isa Distributed) || (arch.ranks[2] == arch.local_index[2]) # The last core
        tripolar_condition = ValidTripolarDomain(condition)
    else # intermediate cores
        tripolar_condition = condition
    end

    return ConditionalOperation(op; func, condition=tripolar_condition, mask)
end

@inline function condition_operand(func, op::TF, condition::AbstractArray, mask)
    arch = architecture(op)
    condition = on_architecture(architecture(operand.grid), condition)

    if !(arch isa Distributed) || (arch.ranks[2] == arch.local_index[2]) # The last core
        tripolar_condition = ValidTripolarDomain(condition)
    else # intermediate cores
        tripolar_condition = condition
    end

    return ConditionalOperation(operand; func, condition=tripolar_condition, mask)
end

# Disambiguation for Immersed reduced fields

# ImmersedReducedFields
const XITRF = AbstractField{Nothing, <:Any, <:Any, <:ITG}
const YITRF = AbstractField{<:Any, Nothing, <:Any, <:ITG}
const ZITRF = AbstractField{<:Any, <:Any, Nothing, <:ITG}

const YZITRF = AbstractField{<:Any, Nothing, Nothing, <:ITG}
const XZITRF = AbstractField{Nothing, <:Any, Nothing, <:ITG}
const XYITRF = AbstractField{Nothing, Nothing, <:Any, <:ITG}

const XYZITRF = AbstractField{Nothing, Nothing, Nothing, <:ITG}

const ITRF = Union{XITRF, YITRF, ZITRF, YZITRF, XZITRF, XYITRF, XYZITRF}

condition_operand(::typeof(identity), op::ITRF, ::Nothing, mask) =
    condition_operand(nothing, op, nothing, mask)
