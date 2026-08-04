using Oceananigans: instantiated_location
using Oceananigans.AbstractOperations: KernelFunctionOperation, Average, Integral, CumulativeIntegral,
                                       Derivative, ∂x, ∂y, ∂z
using Oceananigans.Fields: Scan, reduced_location
using Oceananigans.Operators: interpolation_operator

#####
##### FieldTimeSeriesOperation: AbstractOperations over FieldTimeSeries
#####

struct FieldTimeSeriesOperation{LX, LY, LZ, TI, O, A, P, G, I, χ, T} <: AbstractFieldTimeSeries{LX, LY, LZ, TI, G, T}
    op :: O
    args :: A
    interpolators :: P
    grid :: G
    indices :: I
    times :: χ
    time_indexing :: TI
end

function FieldTimeSeriesOperation{LX, LY, LZ}(op, args::Tuple, interpolators::Tuple, grid,
                                              indices, times, time_indexing, ::Type{T}) where {LX, LY, LZ, T}
    return FieldTimeSeriesOperation{LX, LY, LZ, typeof(time_indexing), typeof(op), typeof(args),
                                    typeof(interpolators), typeof(grid), typeof(indices),
                                    typeof(times), T}(op, args, interpolators, grid, indices,
                                                      times, time_indexing)
end

function GPUAdaptedFieldTimeSeriesOperation{LX, LY, LZ}(op, args::Tuple, interpolators::Tuple, grid,
                                                        indices, times, time_indexing, ::Type{T}) where {LX, LY, LZ, T}
    return GPUAdaptedFieldTimeSeriesOperation{LX, LY, LZ, typeof(time_indexing), typeof(op), typeof(args),
                                              typeof(interpolators), typeof(grid), typeof(indices),
                                              typeof(times), T}(op, args, interpolators, grid, indices,
                                                                times, time_indexing)
end

# Either flavor with the same operator type in parameter position 5, so that each
# pointwise-evaluation body below is written once for the host and GPU-adapted forms.
const SomeFieldTimeSeriesOperation{LX, LY, LZ, TI, O} =
    Union{FieldTimeSeriesOperation{LX, LY, LZ, TI, O},
          GPUAdaptedFieldTimeSeriesOperation{LX, LY, LZ, TI, O}} where {LX, LY, LZ, TI, O}

# Applies `op` at a fixed location when called on time slices. Operator dispatch
# routes here through `AbstractOperations.time_series_operation` with the resolved
# operation location, which @at may specify explicitly.
struct TimeSeriesOperator{L, O}
    loc :: L
    op :: O
end

@inline (tso::TimeSeriesOperator)(args...) = tso.op(tso.loc, args...)

Base.show(io::IO, tso::TimeSeriesOperator) = print(io, tso.op)

Oceananigans.AbstractOperations.time_series_operation(L, op, args...) =
    FieldTimeSeriesOperation(TimeSeriesOperator(L, op), args...)

@inline time_series_operation_argument(a, n) = a
@inline time_series_operation_argument(fts::AbstractFieldTimeSeries, n) = fts[n]

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
    series = Tuple(a for a in args if a isa AbstractFieldTimeSeries)

    isempty(series) &&
        throw(ArgumentError("A FieldTimeSeriesOperation requires at least one FieldTimeSeries argument."))

    times = first(series).times
    time_indexing = first(series).time_indexing

    for fts in series
        fts.times == times ||
            throw(ArgumentError("All FieldTimeSeries arguments of a FieldTimeSeriesOperation must share the same times."))
        fts.time_indexing == time_indexing ||
            throw(ArgumentError("All FieldTimeSeries arguments of a FieldTimeSeriesOperation must share the same time_indexing."))
        fts isa TimeSeriesReductionOperation &&
            throw(ArgumentError("Operating on a lazy Average or Integral of a FieldTimeSeries is not" *
                                " supported; materialize it first with `FieldTimeSeries(reduction)`."))
    end

    # Build a trial slice with the three-dimensional machinery to infer (and validate)
    # the location, grid, indices, and eltype of the operation. Slice at a time index
    # that is already resident so partly-in-memory argument windows do not move.
    trial = time_series_operation_slice(op, args, trial_time_index(first(series)))
    LX, LY, LZ = location(trial)
    grid = Oceananigans.Grids.grid(trial)
    T = eltype(trial)

    # Spatial interpolation operators from each field argument's location to the
    # operation's location (or, for derivatives, the stencil-and-interpolation pair),
    # for pointwise (slice-free) four-dimensional indexing.
    interpolators = pointwise_interpolators(trial, args)

    return FieldTimeSeriesOperation{LX, LY, LZ}(op, args, interpolators, grid, indices(trial),
                                                times, time_indexing, T)
end

trial_time_index(fts::FieldTimeSeries) = fts.backend isa PartlyInMemory ? first(time_indices(fts)) : 1
trial_time_index(operation::FieldTimeSeriesOperation) =
    trial_time_index(first(Tuple(a for a in operation.args if a isa AbstractFieldTimeSeries)))

function pointwise_interpolators(trial, args)
    Lop = instantiated_location(trial)
    return map(a -> a isa AbstractField ? interpolation_operator(instantiated_location(a), Lop) : nothing, args)
end

# A spatial stencil (e.g. a derivative) followed by interpolation to the operation's
# location — the pointwise equivalent of a Derivative slice, constant across time.
struct PointwiseStencil{D, I}
    stencil :: D
    ▶ :: I
end

@inline (ps::PointwiseStencil)(i, j, k, grid, c) = ps.▶(i, j, k, grid, ps.stencil, c)

pointwise_interpolators(trial::Derivative, args) = (PointwiseStencil(trial.∂, trial.▶),)

# Kernel functions index their arguments themselves, and reductions are never evaluated
# pointwise, so neither stores interpolation operators.
pointwise_interpolators(trial::KernelFunctionOperation, args) = map(Returns(nothing), args)
pointwise_interpolators(trial::Scan, args) = map(Returns(nothing), args)

architecture(operation::FieldTimeSeriesOperation) = architecture(operation.grid)

indices(operation::FieldTimeSeriesOperation) = operation.indices
indices(operation::GPUAdaptedFieldTimeSeriesOperation) = operation.indices

# The host form inherits the shared AbstractFieldTimeSeries size.
@inline Base.size(operation::GPUAdaptedFieldTimeSeriesOperation) =
    (size(operation.grid, location(operation), operation.indices)..., length(operation.times))

function Base.summary(operation::FieldTimeSeriesOperation)
    LX, LY, LZ = location(operation)
    return string("FieldTimeSeriesOperation of ", operation.op,
                  " at (", LX, ", ", LY, ", ", LZ, ")",
                  " over ", length(operation.times), " times")
end

Base.show(io::IO, operation::FieldTimeSeriesOperation) = print(io, summary(operation))
Base.show(io::IO, ::MIME"text/plain", operation::FieldTimeSeriesOperation) = show(io, operation)

#####
##### Indexing
#####

Base.getindex(operation::FieldTimeSeriesOperation, n::Int) =
    time_series_operation_slice(operation.op, operation.args, n)

# Pointwise indexing evaluates the operator on pointwise argument values, which is
# equivalent to indexing the slice operation (see the pointwise-evaluation section at
# the bottom of this file). Partly-in-memory arguments need their windows positioned
# first, and on-disk arguments have no resident data at all, so they fall back to
# indexing the slice operation. The branch condition depends only on argument types,
# so it constant-folds.
@propagate_inbounds function Base.getindex(operation::FieldTimeSeriesOperation, i::Int, j::Int, k::Int, n::Int)
    if pointwise_evaluable(operation)
        return pointwise_getindex(operation, i, j, k, n)
    elseif in_memory_arguments(operation)
        update_field_time_series!(operation, n)
        return pointwise_getindex(operation, i, j, k, n)
    else
        return operation[n][i, j, k]
    end
end

# Linear time interpolation of the node values of the operation, reusing the
# machinery that Time-indexes a stored FieldTimeSeries.
@inline Base.getindex(operation::FieldTimeSeriesOperation, i::Int, j::Int, k::Int, time_index::Time) =
    interpolating_getindex(operation, i, j, k, time_index)

# Updating a FieldTimeSeriesOperation updates its FieldTimeSeries arguments;
# other arguments fall through to the no-op fallback. Time-based updates go through
# the shared AbstractFieldTimeSeries method in field_time_series_indexing.jl.
update_field_time_series!(operation::FieldTimeSeriesOperation, n₁::Int, n₂=n₁) =
    foreach(a -> update_field_time_series!(a, n₁, n₂), operation.args)

needs_time_update(operation::FieldTimeSeriesOperation) = any(map(needs_time_update, operation.args))

# Hooks for the shared Time getindex in field_time_series_indexing.jl.
materialized_time_slice(operation::FieldTimeSeriesOperation, n) = compute!(Field(operation[n]))
time_interpolable_slice(operation::FieldTimeSeriesOperation, n) = operation[n]

#####
##### Materialization
#####

"""
$(TYPEDSIGNATURES)

Materialize `operation` into a `FieldTimeSeries` by storing `operation` in the series'
`path` — the provenance that `set!` computes data from, just as file-backed series
compute their data from a file path — and computing the operation at every resident
time index. With the default totally-in-memory backend every time index is computed
once. With `backend = InMemory(N)` only a length-`N` window is resident, and sliding
the window recomputes it from `operation`.

`Time`-indexing the result is identical to `Time`-indexing `operation`.
"""
function FieldTimeSeries(operation::FieldTimeSeriesOperation; backend = InMemory(), kwargs...)
    backend isa OnDisk &&
        throw(ArgumentError("Materializing a FieldTimeSeriesOperation with `backend = OnDisk()` is not" *
                            " supported: the operation itself takes the place of the series' file path."))
    (haskey(kwargs, :path) || haskey(kwargs, :name)) &&
        throw(ArgumentError("The `path` and `name` keyword arguments cannot be used when materializing" *
                            " a FieldTimeSeriesOperation: the operation itself is stored as the path."))

    LX, LY, LZ = location(operation)

    if backend isa PartlyInMemory
        source_windows = map(leaf_time_series(operation)) do source
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

    fts = FieldTimeSeries{LX, LY, LZ}(operation.grid, operation.times;
                                      time_indexing = operation.time_indexing,
                                      path = operation, backend, kwargs...)
    set!(fts)

    return fts
end

# Fill every resident time index of `fts` by computing `operation`. Since
# `set!(fts::InMemoryFTS) = set!(fts, fts.path)`, a series constructed with
# `path = operation` refills its window through this method when the window slides,
# exactly like a file-backed series refills from its file.
function set!(fts::InMemoryFTS, operation::FieldTimeSeriesOperation; kwargs...)
    for n in time_indices(fts)
        set!(fts[n], operation[n])
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
##### Time derivative
#####

struct TimeSeriesTimeDerivative end

const TimeDerivativeOperation = FieldTimeSeriesOperation{<:Any, <:Any, <:Any, <:Any, <:TimeSeriesTimeDerivative}

# Return the finite-difference stencil (n₋, n₊, Δt) for the time derivative at node n:
# centered in the interior, one-sided at the endpoints under Linear and Clamp,
# wrap-aware (and always centered) under Cyclical. Δt is in seconds so that
# DateTime-based times differentiate like number-based times.
@inline function time_derivative_stencil(::Union{Linear, Clamp}, times, n)
    Nt = length(times)
    n₋ = max(n - 1, 1)
    n₊ = min(n + 1, Nt)
    Δt = @inbounds time_difference_seconds(times[n₊], times[n₋])
    return n₋, n₊, Δt
end

@inline function time_derivative_stencil(time_indexing::Cyclical, times, n)
    Nt = length(times)
    n₋ = mod1(n - 1, Nt)
    n₊ = mod1(n + 1, Nt)
    Δt = @inbounds time_difference_seconds(times[n₊], times[n₋])
    T = time_indexing.period
    Δt = ifelse(n₋ > n, Δt + T, ifelse(n₊ < n, Δt + T, Δt))
    return n₋, n₊, Δt
end

"""
$(TYPEDSIGNATURES)

Return the time derivative of `fts` as a `FieldTimeSeriesOperation`, following the same
pattern as every other operation over `FieldTimeSeries`: well-defined node values that
`Time`-indexing linearly interpolates. The node values are centered differences over the
neighboring time nodes in the interior (wrap-aware under `Cyclical` time indexing) and
one-sided differences at the endpoints under `Linear` and `Clamp`:

```jldoctest
using Oceananigans
using Oceananigans.Units: Time

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
times = 0:1.0:2

c = FieldTimeSeries{Center, Center, Center}(grid, times)

for n in 1:length(times)
    set!(c[n], n^2)
end

dc = ∂t(c)

dc[1, 1, 1, 2]

# output
4.0
```
"""
function ∂t(fts::AbstractFieldTimeSeries)
    LX, LY, LZ = location(fts)
    times = fts.times
    time_indexing = fts.time_indexing

    Nt = length(times)
    minimum_length = time_indexing isa Cyclical ? 3 : 2
    Nt < minimum_length &&
        throw(ArgumentError("∂t requires at least $minimum_length times, got $Nt."))

    time_indexing isa Cyclical && isnothing(time_indexing.period) &&
        throw(ArgumentError("∂t with Cyclical time indexing requires an explicitly-specified period."))

    for source in leaf_time_series(fts)
        if source.backend isa PartlyInMemory && length(source.backend) < 4
            throw(ArgumentError("∂t of a partly-in-memory FieldTimeSeries requires a window of at" *
                                " least 4 time indices: Time-interpolating the derivative touches" *
                                " one neighboring node on each side of the bracketing nodes."))
        end
    end

    args = (fts,)
    interpolators = (nothing,) # the argument is colocated with its time derivative

    return FieldTimeSeriesOperation{LX, LY, LZ}(TimeSeriesTimeDerivative(), args, interpolators,
                                                fts.grid, indices(fts), times, time_indexing, eltype(fts))
end

# The slice at node n couples the two neighboring nodes rather than same-index slices.
function Base.getindex(operation::TimeDerivativeOperation, n::Int)
    a = first(operation.args)
    n₋, n₊, Δt = time_derivative_stencil(operation.time_indexing, operation.times, n)
    update_field_time_series!(a, n₋, n₊)
    return (a[n₊] - a[n₋]) / Δt
end

# Updating the derivative at (n₁, n₂) requires the argument one node beyond each side.
function update_field_time_series!(operation::TimeDerivativeOperation, n₁::Int, n₂=n₁)
    a = first(operation.args)
    lo, _ = time_derivative_stencil(operation.time_indexing, operation.times, n₁)
    _, hi = time_derivative_stencil(operation.time_indexing, operation.times, n₂)
    return update_field_time_series!(a, lo, hi)
end

#####
##### GPU adaptation: pointwise, slice-free evaluation inside kernels
#####

function Adapt.adapt_structure(to, operation::FieldTimeSeriesOperation{LX, LY, LZ}) where {LX, LY, LZ}
    return GPUAdaptedFieldTimeSeriesOperation{LX, LY, LZ}(Adapt.adapt(to, operation.op),
                                                          map(a -> Adapt.adapt(to, a), operation.args),
                                                          Adapt.adapt(to, operation.interpolators),
                                                          Adapt.adapt(to, operation.grid),
                                                          Adapt.adapt(to, operation.indices),
                                                          Adapt.adapt(to, operation.times),
                                                          Adapt.adapt(to, operation.time_indexing),
                                                          eltype(operation))
end

function Adapt.adapt_structure(to, kf::TimeSeriesKernelFunction{LX, LY, LZ}) where {LX, LY, LZ}
    func = Adapt.adapt(to, kf.func)
    grid = Adapt.adapt(to, kf.grid)
    return TimeSeriesKernelFunction{LX, LY, LZ, typeof(func), typeof(grid)}(func, grid)
end

function on_architecture(to, operation::FieldTimeSeriesOperation{LX, LY, LZ}) where {LX, LY, LZ}
    return FieldTimeSeriesOperation{LX, LY, LZ}(on_architecture(to, operation.op),
                                                map(a -> on_architecture(to, a), operation.args),
                                                on_architecture(to, operation.interpolators),
                                                on_architecture(to, operation.grid),
                                                on_architecture(to, operation.indices),
                                                on_architecture(to, operation.times),
                                                on_architecture(to, operation.time_indexing),
                                                eltype(operation))
end

function on_architecture(to, kf::TimeSeriesKernelFunction{LX, LY, LZ}) where {LX, LY, LZ}
    grid = on_architecture(to, kf.grid)
    return TimeSeriesKernelFunction{LX, LY, LZ, typeof(kf.func), typeof(grid)}(kf.func, grid)
end

@inline Base.getindex(operation::GPUAdaptedFieldTimeSeriesOperation, i::Int, j::Int, k::Int, n::Int) =
    pointwise_getindex(operation, i, j, k, n)

@inline Base.getindex(operation::GPUAdaptedFieldTimeSeriesOperation, i::Int, j::Int, k::Int, time_index::Time) =
    interpolating_getindex(operation, i, j, k, time_index)

#####
##### Pointwise (slice-free) evaluation
#####

# Building the three-dimensional slice operation on every pointwise access is expensive.
# Pointwise four-dimensional indexing instead spatially interpolates each argument with
# the operators stored at construction and applies the operator to the values —
# equivalent to indexing the slice, with no intermediate slice or Field construction.
# Both requirements below depend only on argument types, so getindex's branch
# constant-folds:
#
# - pointwise_evaluable: no window updates can be triggered by access — every series
#   argument totally in memory.
# - in_memory_arguments: every series argument holds (at least a window of) resident
#   data, so pointwise access is valid once the windows are positioned.
@inline pointwise_evaluable(operation::FieldTimeSeriesOperation) =
    all(map(pointwise_evaluable_argument, operation.args))

@inline pointwise_evaluable_argument(a) = true
@inline pointwise_evaluable_argument(fts::FieldTimeSeries) = fts.backend isa TotallyInMemory
@inline pointwise_evaluable_argument(operation::FieldTimeSeriesOperation) = pointwise_evaluable(operation)

@inline in_memory_arguments(operation::FieldTimeSeriesOperation) =
    all(map(in_memory_argument, operation.args))

@inline in_memory_argument(a) = true
@inline in_memory_argument(fts::FieldTimeSeries) = fts.backend isa AbstractInMemoryBackend
@inline in_memory_argument(operation::FieldTimeSeriesOperation) = in_memory_arguments(operation)

@inline time_series_value(a::SomeTimeSeries, i, j, k, n) = @inbounds a[i, j, k, n]

@inline time_slice(a::SomeTimeSeries, n) = TimeSlice(a, n)
@inline time_slice(a, n) = a

@inline interpolated_argument_value(::Nothing, a, i, j, k, n, grid) = a
@inline interpolated_argument_value(::Nothing, a::SomeTimeSeries, i, j, k, n, grid) = a
@inline interpolated_argument_value(::Nothing, a::AbstractArray, i, j, k, n, grid) = a
@inline interpolated_argument_value(▶, a::SomeTimeSeries, i, j, k, n, grid) = ▶(i, j, k, grid, TimeSlice(a, n))
@inline interpolated_argument_value(▶, a::AbstractArray, i, j, k, n, grid) = ▶(i, j, k, grid, a)

# The raw operator to apply to pointwise argument values: spatial interpolation to the
# operation's location is already done by the stored interpolators, so the location
# carried by a TimeSeriesOperator (which only slicing needs) is stripped.
@inline pointwise_operator(op) = op
@inline pointwise_operator(tso::TimeSeriesOperator) = tso.op

# Pointwise node evaluation dispatches on the operator so that each body is written
# once for the host and GPU-adapted forms.
@propagate_inbounds pointwise_getindex(operation, i, j, k, n) =
    pointwise_getindex(operation.op, operation, i, j, k, n)

@propagate_inbounds pointwise_getindex(op, operation, i, j, k, n) =
    pointwise_operator(op)(map((▶, a) -> interpolated_argument_value(▶, a, i, j, k, n, operation.grid),
                               operation.interpolators, operation.args)...)

# Spatial derivatives evaluate their stencil-and-interpolation pair directly: the
# operator is the stencil, so it must not be re-applied to the resulting value.
const SpatialDerivative = Union{typeof(∂x), typeof(∂y), typeof(∂z)}

@propagate_inbounds pointwise_getindex(::TimeSeriesOperator{<:Any, <:SpatialDerivative}, operation, i, j, k, n) =
    first(operation.interpolators)(i, j, k, operation.grid, TimeSlice(first(operation.args), n))

# Kernel-function form: the kernel function indexes its (time-sliced) arguments itself,
# so arguments at different locations are its responsibility, as for any
# KernelFunctionOperation.
@propagate_inbounds pointwise_getindex(kf::TimeSeriesKernelFunction, operation, i, j, k, n) =
    kf.func(i, j, k, kf.grid, map(a -> time_slice(a, n), operation.args)...)

@propagate_inbounds function pointwise_getindex(::TimeSeriesTimeDerivative, operation, i, j, k, n)
    a = first(operation.args)
    n₋, n₊, Δt = time_derivative_stencil(operation.time_indexing, operation.times, n)
    return (time_series_value(a, i, j, k, n₊) - time_series_value(a, i, j, k, n₋)) / Δt
end

#####
##### Reductions
#####

# Lazy Average, Integral, and CumulativeIntegral of a series: the slice at time index n
# is the Scan over the sliced argument. Scans require a full sweep, so these operations
# are never pointwise evaluable; index them via slices (`Field(op[n])`), interpolate
# globally with `op[Time(t)]`, or materialize them with `FieldTimeSeries(op)`.
struct TimeSeriesReduction{R, K}
    scan :: R
    kwargs :: K
end

(tsr::TimeSeriesReduction)(a) = tsr.scan(a; tsr.kwargs...)

Base.show(io::IO, tsr::TimeSeriesReduction) = print(io, tsr.scan, " with ", tsr.kwargs)

const TimeSeriesReductionOperation = FieldTimeSeriesOperation{<:Any, <:Any, <:Any, <:Any, <:TimeSeriesReduction}

@inline pointwise_evaluable(::TimeSeriesReductionOperation) = false

Base.getindex(operation::TimeSeriesReductionOperation, i::Int, j::Int, k::Int, n::Int) =
    throw(ArgumentError("Pointwise indexing a lazy Average/Integral of a FieldTimeSeries is not" *
                        " supported: reductions require a full sweep. Materialize a slice with" *
                        " `Field(operation[n])`, interpolate globally with `operation[Time(t)]`," *
                        " or materialize the series with `FieldTimeSeries(operation)`."))

Adapt.adapt_structure(to, operation::FieldTimeSeriesOperation{LX, LY, LZ, TI, <:TimeSeriesReduction}) where {LX, LY, LZ, TI} =
    throw(ArgumentError("A lazy Average/Integral of a FieldTimeSeries cannot be used inside GPU" *
                        " kernels; materialize it first with `FieldTimeSeries(operation)`."))

# Reduction slices interpolate in time (and feed outer reductions) as computed Fields.
time_interpolable_slice(operation::TimeSeriesReductionOperation, n) = compute!(Field(operation[n]))
reduction_time_slice(operation::TimeSeriesReductionOperation, n) = compute!(Field(operation[n]))

# Scans cannot be broadcast onto a Field, so materialize reduced slices via `Field`,
# reusing one buffer across time indices.
function set!(fts::InMemoryFTS, operation::TimeSeriesReductionOperation; kwargs...)
    temp = nothing
    for n in time_indices(fts)
        temp = isnothing(temp) ? Field(operation[n]) : Field(operation[n]; data=temp.data)
        set!(fts[n], temp)
    end
    return fts
end

Oceananigans.AbstractOperations.Average(fts::AbstractFieldTimeSeries; kwargs...) =
    FieldTimeSeriesOperation(TimeSeriesReduction(Average, NamedTuple(kwargs)), fts)

Oceananigans.AbstractOperations.Integral(fts::AbstractFieldTimeSeries; kwargs...) =
    FieldTimeSeriesOperation(TimeSeriesReduction(Integral, NamedTuple(kwargs)), fts)

Oceananigans.AbstractOperations.CumulativeIntegral(fts::AbstractFieldTimeSeries; kwargs...) =
    FieldTimeSeriesOperation(TimeSeriesReduction(CumulativeIntegral, NamedTuple(kwargs)), fts)

# Hooks for the shared reduction methods in field_time_series_reductions.jl: slices of
# a generic operation reduce lazily (reductions apply pointwise without materializing),
# and accumulate through a reusable temporary Field.
reduction_time_slice(operation::FieldTimeSeriesOperation, n) = operation[n]

materialized_time_slice!(temp, operation::FieldTimeSeriesOperation, n) = set!(temp, operation[n])
materialized_time_slice!(temp, operation::TimeSeriesReductionOperation, n) = Field(operation[n]; data=temp.data)

Oceananigans.Fields.conditional_length(operation::FieldTimeSeriesOperation) = prod(size(operation))
