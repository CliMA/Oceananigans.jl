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
                                    co::ConditionalOperation, args...)
    Nx, Ny, _ = size(grid)

    Ly = Oceananigans.Fields.location(co)[2]
    TY = Oceananigans.Grids.topology(grid, 2)

    last_half_row  = ((i > Nx÷2) & (j == Ny))
    full_final_row = (j == Ny)

    face_folded_domain   = ifelse(Ly == Face, !last_half_row, !full_final_row)
    center_folded_domain = ifelse(Ly == Face, true,           !last_half_row)

    return ifelse(TY == RightFaceFolded, face_folded_domain, center_folded_domain)
end

@inline function evaluate_condition(vd::PrognosticTripolarCells,
                                    i, j, k,
                                    grid::TripolarGridOfSomeKind,
                                    co::ConditionalOperation, args...)

    prognostic_domain = evaluate_condition(PrognosticTripolarCells(), i, j, k, grid, co)
    return prognostic_domain & evaluate_condition(vd.condition, i, j, k, grid, co, args...)
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

@inline function tripolar_condition_operand(func, op, condition, mask)
    arch = architecture(op)

    if !(arch isa Distributed) || (arch.ranks[2] == arch.local_index[2]) # The last core
        tripolar_condition = PrognosticTripolarCells()
    else # intermediate cores
        tripolar_condition = nothing
    end

    return ConditionalOperation(op; func, condition=tripolar_condition, mask)
end

@inline condition_operand(::typeof(identity), op::TF, ::Nothing, mask) = tripolar_condition_operand(nothing, op, nothing, mask)
@inline condition_operand(::Nothing, op::TF, ::Nothing, mask) = tripolar_condition_operand(nothing, op, nothing, mask)
@inline condition_operand(func, op::TF, condition, mask) = tripolar_condition_operand(func, op, condition, mask)

@inline function condition_operand(func, op::TF, condition::AbstractArray, mask)
    arch = architecture(op)
    condition = on_architecture(arch, condition)
    return tripolar_condition_operand(func, op, condition, mask)
end

# Disambiguation for Immersed Tripolar Fields (ITF) and Immersed Tripolar Reduced Fields (ITRF)
# These types match both IF/IRF (from ImmersedBoundaries) and TF (from tripolar reductions)

using Oceananigans.ImmersedBoundaries: NotImmersedColumn, immersed_column

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

function immersed_reduced_tripolar_condition_operand(func, op, condition, mask)
    arch = architecture(op)
    immersed_cond = NotImmersedColumn(immersed_column(op), condition)

    if !(arch isa Distributed) || (arch.ranks[2] == arch.local_index[2]) # The last core
        tripolar_condition = PrognosticTripolarCells(immersed_cond)
    else # intermediate cores
        tripolar_condition = immersed_cond
    end

    return ConditionalOperation(op; func, condition=tripolar_condition, mask)
end

@inline condition_operand(::typeof(identity), op::ITRF, ::Nothing, mask) = immersed_reduced_tripolar_condition_operand(nothing, op, nothing, mask)
@inline condition_operand(::Nothing, op::ITRF, ::Nothing, mask) = immersed_reduced_tripolar_condition_operand(nothing, op, nothing, mask)
@inline condition_operand(func, op::ITRF, condition, mask) = immersed_reduced_tripolar_condition_operand(func, op, condition, mask)

@inline function condition_operand(func, op::ITRF, condition::AbstractArray, mask)
    arch = architecture(op)
    condition = on_architecture(architecture(op.grid), condition)
    return immersed_reduced_tripolar_condition_operand(func, op, condition, mask)
end
