#####
##### FieldTimeSeriesOperation: AbstractOperations over FieldTimeSeries
#####

struct FieldTimeSeriesOperation{LX, LY, LZ, TI, O, A, G, χ, T} <: AbstractField{LX, LY, LZ, G, T, 4}
    op :: O
    args :: A
    grid :: G
    times :: χ
    time_indexing :: TI
end

const FieldTimeSeriesLike = Union{FieldTimeSeries, FieldTimeSeriesOperation}

@inline time_series_operation_argument(a, n) = a
@inline time_series_operation_argument(fts::FieldTimeSeriesLike, n) = fts[n]

"""
$(TYPEDSIGNATURES)

Return the three-dimensional operation `op(args...)` at time index `n`, slicing
`FieldTimeSeries` (and `FieldTimeSeriesOperation`) arguments at `n` and passing
other arguments through unchanged.
"""
time_series_operation_slice(op, args, n) =
    op(map(a -> time_series_operation_argument(a, n), args)...)

"""
$(TYPEDSIGNATURES)

Return a lazy representation of the operator `op` applied at every time node to `args`,
at least one of which is a `FieldTimeSeries` or another `FieldTimeSeriesOperation`.
All `FieldTimeSeries` arguments must share the same `times` and `time_indexing`.

`FieldTimeSeriesOperation`s are usually built by applying ordinary operators to
`FieldTimeSeries`, the same way `AbstractOperation`s are built from `Field`s:

```jldoctest fieldtimeseriesoperation
using Oceananigans
using Oceananigans.Units: Time

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
times = 0:1.0:2

a = FieldTimeSeries{Center, Center, Center}(grid, times)
b = FieldTimeSeries{Center, Center, Center}(grid, times)

for n in 1:length(times)
    set!(a[n], n)
    set!(b[n], 2n)
end

ab = a * b

ab[1, 1, 1, 2]

# output
8.0
```

Indexed at time index `n`, the operation returns the three-dimensional `AbstractOperation`
over the sliced arguments, which can be `compute!`d or indexed like any other operation.

Indexed at `Time(t)`, the operation linearly interpolates between the two time-node
values of the *result* — the same semantics as `Time`-indexing a `FieldTimeSeries`
that stores `op(args...)` computed at every node:

```jldoctest fieldtimeseriesoperation
ab[1, 1, 1, Time(0.5)]

# output
5.0
```

Note that for a nonlinear operator this differs (between nodes) from applying the
operator to `Time`-interpolated arguments: `Time`-indexing above returns
`(1 * 2 + 2 * 4) / 2 = 5`, while `a[1, 1, 1, Time(0.5)] * b[1, 1, 1, Time(0.5)] = 1.5 * 3 = 4.5`.
For the latter semantics, wrap the arguments in `TimeSeriesInterpolation` instead.
"""
function FieldTimeSeriesOperation(op, args...)
    series = Tuple(a for a in args if a isa FieldTimeSeriesLike)

    isempty(series) &&
        throw(ArgumentError("A FieldTimeSeriesOperation requires at least one FieldTimeSeries argument."))

    times = first(series).times
    time_indexing = first(series).time_indexing

    for fts in series
        fts.times == times ||
            throw(ArgumentError("All FieldTimeSeries arguments of a FieldTimeSeriesOperation must share the same times."))
        fts.time_indexing == time_indexing ||
            throw(ArgumentError("All FieldTimeSeries arguments of a FieldTimeSeriesOperation must share the same time_indexing."))
    end

    # Build a trial slice with the three-dimensional machinery to infer (and validate)
    # the location, grid, and eltype of the operation.
    trial = time_series_operation_slice(op, args, 1)
    LX, LY, LZ = location(trial)
    grid = trial.grid
    T = eltype(trial)

    return FieldTimeSeriesOperation{LX, LY, LZ, typeof(time_indexing), typeof(op), typeof(args),
                                    typeof(grid), typeof(times), T}(op, args, grid, times, time_indexing)
end

architecture(fts_op::FieldTimeSeriesOperation) = architecture(fts_op.grid)

indices(fts_op::FieldTimeSeriesOperation) = indices(fts_op[1])

@inline Base.size(fts_op::FieldTimeSeriesOperation) =
    (size(fts_op.grid, location(fts_op), indices(fts_op))..., length(fts_op.times))

function Base.summary(fts_op::FieldTimeSeriesOperation)
    LX, LY, LZ = location(fts_op)
    return string("FieldTimeSeriesOperation of ", fts_op.op,
                  " at (", LX, ", ", LY, ", ", LZ, ")",
                  " over ", length(fts_op.times), " times")
end

Base.show(io::IO, fts_op::FieldTimeSeriesOperation) = print(io, summary(fts_op))
Base.show(io::IO, ::MIME"text/plain", fts_op::FieldTimeSeriesOperation) = show(io, fts_op)

#####
##### Indexing
#####

Base.getindex(fts_op::FieldTimeSeriesOperation, n::Int) =
    time_series_operation_slice(fts_op.op, fts_op.args, n)

@propagate_inbounds Base.getindex(fts_op::FieldTimeSeriesOperation, i::Int, j::Int, k::Int, n::Int) =
    fts_op[n][i, j, k]

# Linear time interpolation of the node values of the operation, reusing the
# machinery that Time-indexes a stored FieldTimeSeries.
@inline Base.getindex(fts_op::FieldTimeSeriesOperation, i::Int, j::Int, k::Int, time_index::Time) =
    interpolating_getindex(fts_op, i, j, k, time_index)

# TODO: support partly-in-memory FieldTimeSeries arguments, which require both
# bracketing time indices to be resident simultaneously (cf. update_field_time_series!).
function Base.getindex(fts_op::FieldTimeSeriesOperation, time_index::Time)
    interpolator = cpu_interpolating_time_indices(architecture(fts_op), fts_op.times,
                                                  fts_op.time_indexing, time_index.time)
    ñ = interpolator.fractional_index
    n₁ = interpolator.first_index
    n₂ = interpolator.second_index

    n₁ == n₂ && return compute!(Field(fts_op[n₁]))

    t₂ = @allowscalar fts_op.times[n₂]
    t₁ = @allowscalar fts_op.times[n₁]
    t = interp_time(t₁, t₂, ñ)
    status = FixedTime(t)

    ψ̃ = Field(fts_op[n₂] * ñ + fts_op[n₁] * (1 - ñ); status)

    return compute!(ψ̃)
end

#####
##### Materialization
#####

"""
$(TYPEDSIGNATURES)

Materialize `fts_op` into a `FieldTimeSeries` by computing the operation at every
time node. `Time`-indexing the result is identical to `Time`-indexing `fts_op`.
"""
function FieldTimeSeries(fts_op::FieldTimeSeriesOperation; kwargs...)
    LX, LY, LZ = location(fts_op)
    fts = FieldTimeSeries{LX, LY, LZ}(fts_op.grid, fts_op.times;
                                      time_indexing = fts_op.time_indexing, kwargs...)

    fts.backend isa TotallyInMemory ||
        throw(ArgumentError("Materializing a FieldTimeSeriesOperation requires a totally in-memory backend."))

    set!(fts, fts_op)

    return fts
end

function set!(fts::FieldTimeSeries, fts_op::FieldTimeSeriesOperation)
    for n in time_indices(fts.backend, fts.time_indexing, length(fts.times))
        set!(fts[n], fts_op[n])
    end
    return fts
end

#####
##### Operators: the same operations that build AbstractOperations from Fields
##### build FieldTimeSeriesOperations from FieldTimeSeries.
#####

for op in (:+, :-, :*, :/, :^)
    @eval begin
        Base.$op(a::FieldTimeSeriesLike, b::FieldTimeSeriesLike) = FieldTimeSeriesOperation(Base.$op, a, b)
        Base.$op(a::FieldTimeSeriesLike, b::AbstractField) = FieldTimeSeriesOperation(Base.$op, a, b)
        Base.$op(a::AbstractField, b::FieldTimeSeriesLike) = FieldTimeSeriesOperation(Base.$op, a, b)
        Base.$op(a::FieldTimeSeriesLike, b::Number) = FieldTimeSeriesOperation(Base.$op, a, b)
        Base.$op(a::Number, b::FieldTimeSeriesLike) = FieldTimeSeriesOperation(Base.$op, a, b)
    end
end

for op in (:sqrt, :sin, :cos, :exp, :tanh, :abs, :log10, :log, :tan, :sinh, :cosh)
    @eval Base.$op(a::FieldTimeSeriesLike) = FieldTimeSeriesOperation(Base.$op, a)
end

Base.:-(a::FieldTimeSeriesLike) = FieldTimeSeriesOperation(Base.:-, a)
