using Oceananigans.AbstractOperations: KernelFunctionOperation

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

# Updating a FieldTimeSeriesOperation updates its FieldTimeSeries arguments;
# other arguments fall through to the no-op fallback.
update_field_time_series!(fts_op::FieldTimeSeriesOperation, n₁::Int, n₂=n₁) =
    foreach(a -> update_field_time_series!(a, n₁, n₂), fts_op.args)

function update_field_time_series!(fts_op::FieldTimeSeriesOperation, time_index::Time)
    interpolator = cpu_interpolating_time_indices(architecture(fts_op), fts_op.times,
                                                  fts_op.time_indexing, time_index.time)
    return update_field_time_series!(fts_op, interpolator.first_index, interpolator.second_index)
end

function Base.getindex(fts_op::FieldTimeSeriesOperation, time_index::Time)
    interpolator = cpu_interpolating_time_indices(architecture(fts_op), fts_op.times,
                                                  fts_op.time_indexing, time_index.time)
    ñ = interpolator.fractional_index
    n₁ = interpolator.first_index
    n₂ = interpolator.second_index

    n₁ == n₂ && return compute!(Field(fts_op[n₁]))

    # Ensure both bracketing time indices of partly-in-memory arguments are
    # resident simultaneously before slicing.
    update_field_time_series!(fts_op, n₁, n₂)

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

Materialize `fts_op` into a `FieldTimeSeries` by storing `fts_op` in the series'
`path` — the provenance that `set!` computes data from, just as file-backed series
compute their data from a file path — and computing the operation at every resident
time index. With the default totally-in-memory backend every time index is computed
once. With `backend = InMemory(N)` only a length-`N` window is resident, and sliding
the window recomputes it from `fts_op`.

`Time`-indexing the result is identical to `Time`-indexing `fts_op`.
"""
function FieldTimeSeries(fts_op::FieldTimeSeriesOperation; backend = InMemory(), kwargs...)
    LX, LY, LZ = location(fts_op)

    if backend isa PartlyInMemory
        source_windows = map(extract_field_time_series(fts_op)) do source
            window = time_indices_length(source.backend, source.times)
            isnothing(window) ? length(backend) : window
        end

        if !isempty(source_windows) && minimum(source_windows) < length(backend)
            @warn string("A FieldTimeSeries argument holds a window of ", minimum(source_windows),
                         " time indices, shorter than the materialized window of ",
                         length(backend), ": every window slide will reload the",
                         " argument's window repeatedly.")
        end
    end

    fts = FieldTimeSeries{LX, LY, LZ}(fts_op.grid, fts_op.times;
                                      time_indexing = fts_op.time_indexing,
                                      path = fts_op, backend, kwargs...)
    set!(fts)

    return fts
end

# Fill every resident time index of `fts` by computing `fts_op`. Since
# `set!(fts::InMemoryFTS) = set!(fts, fts.path)`, a series constructed with
# `path = fts_op` refills its window through this method when the window slides,
# exactly like a file-backed series refills from its file.
function set!(fts::InMemoryFTS, fts_op::FieldTimeSeriesOperation; kwargs...)
    for n in time_indices(fts)
        set!(fts[n], fts_op[n])
    end
    return fts
end

#####
##### KernelFunctionOperation over FieldTimeSeries
#####

# Callable that builds the KernelFunctionOperation for one time slice.
struct TimeSeriesKernelFunction{LX, LY, LZ, F, G}
    func :: F
    grid :: G
end

@inline (kf::TimeSeriesKernelFunction{LX, LY, LZ})(args...) where {LX, LY, LZ} =
    KernelFunctionOperation{LX, LY, LZ}(kf.func, kf.grid, args...)

Base.show(io::IO, kf::TimeSeriesKernelFunction) =
    print(io, "KernelFunctionOperation of ", kf.func)

"""
$(TYPEDSIGNATURES)

Return a `FieldTimeSeriesOperation` at location `(LX, LY, LZ)` on `grid` whose slice at
time index `n` is `KernelFunctionOperation{LX, LY, LZ}(func, grid, args_at_n...)`, where
`FieldTimeSeries` arguments are sliced at `n` and other arguments (e.g. parameters)
pass through unchanged:

```jldoctest
using Oceananigans

@inline scaled_product(i, j, k, grid, a, b, scale) = @inbounds scale * a[i, j, k] * b[i, j, k]

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
times = 0:1.0:2

a = FieldTimeSeries{Center, Center, Center}(grid, times)
b = FieldTimeSeries{Center, Center, Center}(grid, times)

for n in 1:length(times)
    set!(a[n], n)
    set!(b[n], 2n)
end

q = FieldTimeSeriesOperation{Center, Center, Center}(scaled_product, grid, a, b, 10)

q[1, 1, 1, 2]

# output
80.0
```
"""
function FieldTimeSeriesOperation{LX, LY, LZ}(func, grid::AbstractGrid, args...) where {LX, LY, LZ}
    kf = TimeSeriesKernelFunction{LX, LY, LZ, typeof(func), typeof(grid)}(func, grid)
    return FieldTimeSeriesOperation(kf, args...)
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
