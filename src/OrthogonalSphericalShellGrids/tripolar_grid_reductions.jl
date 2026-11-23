using Oceananigans.AbstractOperations: ConditionalOperation, evaluate_condition, validate_condition
using Oceananigans.Fields: AbstractField, OneField

#####
##### Reductions on tripolar grids count each cell of the fold row once:
##### the zipper overwrites the fold-row cell `i` with its mirror `i′` when `i > i′`.
#####

struct PrognosticTripolarCells{F} <: Function
    condition :: F
    i₀ :: Int # global x-offset of the local grid
    Nx :: Int # global x-size
end

function PrognosticTripolarCells(condition, grid)
    Nx = size(grid, 1)
    rx, Rx = x_rank(architecture(grid))
    return PrognosticTripolarCells(condition, (rx - 1) * Nx, Rx * Nx)
end

x_rank(arch) = (1, 1)
x_rank(arch::Distributed) = (arch.local_index[1], ranks(arch)[1])

Base.summary(::PrognosticTripolarCells{Nothing}) = "PrognosticTripolarCells()"
Base.summary(vd::PrognosticTripolarCells) = string("PrognosticTripolarCells(", summary(vd.condition), ")")
Base.size(vd::PrognosticTripolarCells{<:AbstractArray}) = size(vd.condition)

AbstractOperations.validate_condition(cond::PrognosticTripolarCells{<:AbstractArray}, ::OneField) = cond
AbstractOperations.validate_condition(cond::PrognosticTripolarCells{<:AbstractArray}, operand::AbstractField) = validate_condition(cond.condition, operand)

"Adapt `PrognosticTripolarCells` to work on the GPU via KernelAbstractions."
Adapt.adapt_structure(to, vd::PrognosticTripolarCells) = PrognosticTripolarCells(Adapt.adapt(to, vd.condition), vd.i₀, vd.Nx)

@inline function fold_row_copy(i, j, cells, grid, co)
    LX, LY, _ = Oceananigans.Fields.location(co)
    TY = topology(grid, 2)
    Ny = size(grid, 2)

    fold_row = ((TY <: UPivotTopology) & (LY == Center) & (j == Ny)) |
               ((TY <: FPivotTopology) & (LY == Face)   & (j == Ny + 1))

    Nx = cells.Nx
    i += cells.i₀
    i′ = mod1(Nx - i + 1 + (LX == Face) + pivot_shift(fold_pivot(grid)), Nx)

    return fold_row & (LX !== Nothing) & (i > i′) # an x-reduced field holds one value per row
end

@inline AbstractOperations.evaluate_condition(cells::PrognosticTripolarCells{Nothing}, i, j, k,
                                              grid::TripolarGridOfSomeKind, co::ConditionalOperation, args...) =
    !fold_row_copy(i, j, cells, grid, co)

@inline AbstractOperations.evaluate_condition(cells::PrognosticTripolarCells, i, j, k,
                                              grid::TripolarGridOfSomeKind, co::ConditionalOperation, args...) =
    !fold_row_copy(i, j, cells, grid, co) & evaluate_condition(cells.condition, i, j, k, grid, co, args...)

#####
##### conditional_operand extension
#####

ITG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}

const TF = Union{<:AbstractField{<:Any, <:Any, <:Any, <:TripolarGridOfSomeKind},
                 <:AbstractField{<:Any, <:Any, <:Any, <:ITG}}

@inline Fields.conditional_length(c::TF) = sum(conditional_one(c, 0))
@inline Fields.conditional_length(c::TF, dims::Int) = sum(conditional_one(c, 0); dims)
@inline Fields.conditional_length(c::TF, dims::Tuple) = sum(conditional_one(c, 0); dims)

@inline tripolar_condition_operand(func, op, condition, mask) =
    ConditionalOperation(op; func, condition=PrognosticTripolarCells(condition, op.grid), mask)

@inline Fields.condition_operand(::typeof(identity), op::TF, ::Nothing, mask) = tripolar_condition_operand(nothing, op, nothing, mask)
@inline Fields.condition_operand(::Nothing, op::TF, ::Nothing, mask) = tripolar_condition_operand(nothing, op, nothing, mask)
@inline Fields.condition_operand(func, op::TF, condition, mask) = tripolar_condition_operand(func, op, condition, mask)

@inline function Fields.condition_operand(func, op::TF, condition::AbstractArray, mask)
    arch = architecture(op)
    condition = on_architecture(arch, condition)
    return tripolar_condition_operand(func, op, condition, mask)
end

const TripolarConditionalOperation = ConditionalOperation{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:TripolarGridOfSomeKind}

@inline Fields.condition_operand(::Nothing, op::TripolarConditionalOperation, ::Nothing, mask) = op
@inline Fields.condition_operand(::typeof(identity), op::TripolarConditionalOperation, ::Nothing, mask) = op

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
    immersed_condition = NotImmersedColumn(immersed_column(op), condition)

    # `immersed_condition` already masks the immersed columns
    LX, LY, LZ = Fields.location(op)
    return ConditionalOperation{LX, LY, LZ}(op, func, op.grid, PrognosticTripolarCells(immersed_condition, op.grid), mask)
end

@inline Fields.condition_operand(::typeof(identity), op::ITRF, ::Nothing, mask) = immersed_reduced_tripolar_condition_operand(nothing, op, nothing, mask)
@inline Fields.condition_operand(::Nothing, op::ITRF, ::Nothing, mask) = immersed_reduced_tripolar_condition_operand(nothing, op, nothing, mask)
@inline Fields.condition_operand(func, op::ITRF, condition, mask) = immersed_reduced_tripolar_condition_operand(func, op, condition, mask)

@inline function Fields.condition_operand(func, op::ITRF, condition::AbstractArray, mask)
    arch = architecture(op)
    condition = on_architecture(architecture(op.grid), condition)
    return immersed_reduced_tripolar_condition_operand(func, op, condition, mask)
end
