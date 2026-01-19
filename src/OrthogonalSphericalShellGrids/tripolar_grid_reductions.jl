using Oceananigans.AbstractOperations: ConditionalOperation
using Oceananigans.Fields: AbstractField, OneField, instantiated_location

import Oceananigans.AbstractOperations: evaluate_condition, validate_condition
import Oceananigans.Fields: condition_operand, conditional_length

#####
##### Reduction operations involving tripolar grids exclude the repeated row at the top
##### of the domain for Fields located on `Center`s in meridional direction.
#####

struct PrognosticTripolarCells{F} <: Function
    condition :: F
end

PrognosticTripolarCells() = PrognosticTripolarCells(nothing)

Base.summary(::PrognosticTripolarCells{Nothing}) = "PrognosticTripolarCells()"
Base.summary(vd::PrognosticTripolarCells) = string("PrognosticTripolarCells(", summary(vd.condition), ")")
Base.size(vd::PrognosticTripolarCells{<:AbstractArray}) = size(vd.condition)

validate_condition(cond::PrognosticTripolarCells{<:AbstractArray}, ::OneField) = cond
validate_condition(cond::PrognosticTripolarCells{<:AbstractArray}, operand::AbstractField) = validate_condition(cond.condition, operand)

"Adapt `PrognosticTripolarCells` to work on the GPU via KernelAbstractions."
Adapt.adapt_structure(to, vd::PrognosticTripolarCells) = PrognosticTripolarCells(Adapt.adapt(to, vd.condition))

@inline function evaluate_condition(::PrognosticTripolarCells{Nothing},
                                    i, j, k,
                                    grid::TripolarGridOfSomeKind,
                                    co::ConditionalOperation) #, args...)
    ℓx, ℓy, ℓz = Oceananigans.Fields.location(co)
    Nx, Ny, _ = size(grid)
    valid_domain = !((i > Nx÷2) & (j == Ny))
    return ifelse(ℓy == Face, true, valid_domain)
end

@inline function evaluate_condition(vd::PrognosticTripolarCells,
                                    i, j, k,
                                    grid::TripolarGridOfSomeKind,
                                    co::ConditionalOperation, args...)

    valid_domain = evaluate_condition(PrognosticTripolarCells(), i, j, k, grid, co)
    return valid_domain & evaluate_condition(vd.condition, i, j, k, grid, co, args...)
end

#####
##### conditional_operand extension
#####

ITG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}

const TF = Union{<:AbstractField{<:Any, <:Any, <:Any, <:TripolarGridOfSomeKind},
                 <:AbstractField{<:Any, <:Any, <:Any, <:ITG}}

@inline conditional_length(c::TF) = conditional_length(condition_operand(identity, c, PrognosticTripolarCells(), 0))
@inline conditional_length(c::TF, ::Colon) = conditional_length(c)
@inline conditional_length(c::TF, ::NTuple{3}) = conditional_length(c)
@inline conditional_length(c::TF, d::Int) = conditional_length(condition_operand(identity, c, PrognosticTripolarCells(), 0), d)
@inline conditional_length(c::TF, dims::NTuple{1}) = conditional_length(c, dims[1])
@inline conditional_length(c::TF, dims::NTuple{2}) = conditional_length(condition_operand(identity, c, PrognosticTripolarCells(), 0), dims)

condition_operand(::typeof(identity), op::TF, ::Nothing, mask) =
    condition_operand(nothing, op, nothing, mask)

@inline function condition_operand(::Nothing, op::TF, ::Nothing, mask)
    arch = architecture(op)

    if !(arch isa Distributed) || (arch.ranks[2] == arch.local_index[2]) # The last core
        tripolar_condition = PrognosticTripolarCells()
    else # intermediate cores
        tripolar_condition = nothing
    end

    return ConditionalOperation(op; func=nothing, condition=tripolar_condition, mask)
end

@inline function condition_operand(func, op::TF, condition, mask)
    arch = architecture(op)

    if !(arch isa Distributed) || (arch.ranks[2] == arch.local_index[2]) # The last core
        tripolar_condition = PrognosticTripolarCells(condition)
    else # intermediate cores
        tripolar_condition = condition
    end

    return ConditionalOperation(op; func, condition=tripolar_condition, mask)
end

@inline function condition_operand(func, op::TF, condition::AbstractArray, mask)
    arch = architecture(op)
    condition = on_architecture(architecture(operand.grid), condition)

    if !(arch isa Distributed) || (arch.ranks[2] == arch.local_index[2]) # The last core
        tripolar_condition = PrognosticTripolarCells(condition)
    else # intermediate cores
        tripolar_condition = condition
    end

    return ConditionalOperation(operand; func, condition=tripolar_condition, mask)
end

# Disambiguation for Immersed Tripolar Fields (ITF) and Immersed Tripolar Reduced Fields (ITRF)
# These types match both IF/IRF (from ImmersedBoundaries) and TF (from tripolar reductions)

using Oceananigans.ImmersedBoundaries: NotImmersed, NotImmersedColumn, immersed_column

# Immersed Tripolar Fields (non-reduced)
const ITF = AbstractField{<:Any, <:Any, <:Any, <:ITG}

# Immersed Tripolar Reduced Fields
const XITRF = AbstractField{Nothing, <:Any, <:Any, <:ITG}
const YITRF = AbstractField{<:Any, Nothing, <:Any, <:ITG}
const ZITRF = AbstractField{<:Any, <:Any, Nothing, <:ITG}

const YZITRF = AbstractField{<:Any, Nothing, Nothing, <:ITG}
const XZITRF = AbstractField{Nothing, <:Any, Nothing, <:ITG}
const XYITRF = AbstractField{Nothing, Nothing, <:Any, <:ITG}

const XYZITRF = AbstractField{Nothing, Nothing, Nothing, <:ITG}

const ITRF = Union{XITRF, YITRF, ZITRF, YZITRF, XZITRF, XYITRF, XYZITRF}

# Disambiguation: ITRF matches both IRF and TF, so we need explicit methods
# For ITRF, we combine both tripolar (PrognosticTripolarCells) and immersed (NotImmersedColumn) conditions

condition_operand(::typeof(identity), op::ITRF, ::Nothing, mask) =
    condition_operand(nothing, op, nothing, mask)

@inline function condition_operand(::Nothing, op::ITRF, ::Nothing, mask)
    arch = architecture(op)
    immersed_cond = NotImmersedColumn(immersed_column(op), nothing)

    if !(arch isa Distributed) || (arch.ranks[2] == arch.local_index[2]) # The last core
        tripolar_condition = PrognosticTripolarCells(immersed_cond)
    else # intermediate cores
        tripolar_condition = immersed_cond
    end

    return ConditionalOperation(op; func=nothing, condition=tripolar_condition, mask)
end

@inline function condition_operand(func, op::ITRF, condition, mask)
    arch = architecture(op)
    immersed_cond = NotImmersedColumn(immersed_column(op), condition)

    if !(arch isa Distributed) || (arch.ranks[2] == arch.local_index[2]) # The last core
        tripolar_condition = PrognosticTripolarCells(immersed_cond)
    else # intermediate cores
        tripolar_condition = immersed_cond
    end

    return ConditionalOperation(op; func, condition=tripolar_condition, mask)
end

@inline function condition_operand(func, op::ITRF, condition::AbstractArray, mask)
    arch = architecture(op)
    condition = on_architecture(architecture(op.grid), condition)
    immersed_cond = NotImmersedColumn(immersed_column(op), condition)

    if !(arch isa Distributed) || (arch.ranks[2] == arch.local_index[2]) # The last core
        tripolar_condition = PrognosticTripolarCells(immersed_cond)
    else # intermediate cores
        tripolar_condition = immersed_cond
    end

    return ConditionalOperation(op; func, condition=tripolar_condition, mask)
end
