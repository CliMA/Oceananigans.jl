using Statistics: Statistics, mean

using Oceananigans: instantiated_location
using Oceananigans.AbstractOperations: KernelFunctionOperation, Average, Integral, Derivative, ∂x, ∂y, ∂z
using Oceananigans.Fields: Scan, reduced_location
using Oceananigans.Operators: interpolation_operator

#####
##### FieldTimeSeriesOperation: AbstractOperations over FieldTimeSeries
#####

struct FieldTimeSeriesOperation{LX, LY, LZ, TI, O, A, P, G, χ, T} <: AbstractFieldTimeSeries{LX, LY, LZ, TI, G, T}
    op :: O
    args :: A
    interpolators :: P
    grid :: G
    times :: χ
    time_indexing :: TI
end

const FieldTimeSeriesLike = AbstractFieldTimeSeries

# The grid an operation acts on; Scans (reductions like Average) carry it on their operand.
operation_grid(a) = a.grid
operation_grid(scan::Scan) = operation_grid(scan.operand)

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
    grid = operation_grid(trial)
    T = eltype(trial)

    # Spatial interpolation operators from each field argument's location to the
    # operation's location (or, for derivatives, the stencil-and-interpolation pair),
    # for pointwise (slice-free) four-dimensional indexing.
    interpolators = pointwise_interpolators(trial, args)

    return FieldTimeSeriesOperation{LX, LY, LZ, typeof(time_indexing), typeof(op), typeof(args), typeof(interpolators),
                                    typeof(grid), typeof(times), T}(op, args, interpolators, grid, times, time_indexing)
end

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

# Pointwise indexing evaluates the operator on pointwise argument values when that is
# provably equivalent (see pointwise_evaluable at the bottom of this file); otherwise
# it falls back to indexing the slice operation, which spatially interpolates arguments
# and updates partly-in-memory windows.
@propagate_inbounds function Base.getindex(fts_op::FieldTimeSeriesOperation, i::Int, j::Int, k::Int, n::Int)
    if pointwise_evaluable(fts_op) # constant-folds: depends only on argument types
        return pointwise_getindex(fts_op, i, j, k, n)
    else
        return fts_op[n][i, j, k]
    end
end

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
##### Time derivative
#####

struct TimeSeriesTimeDerivative end

const TimeDerivativeFTSOp = FieldTimeSeriesOperation{<:Any, <:Any, <:Any, <:Any, <:TimeSeriesTimeDerivative}

# Return the finite-difference stencil (n₋, n₊, Δt) for the time derivative at node n:
# centered in the interior, one-sided at the endpoints under Linear and Clamp,
# wrap-aware (and always centered) under Cyclical.
@inline function time_derivative_stencil(::Union{Linear, Clamp}, times, n)
    Nt = length(times)
    n₋ = max(n - 1, 1)
    n₊ = min(n + 1, Nt)
    Δt = @inbounds times[n₊] - times[n₋]
    return n₋, n₊, Δt
end

@inline function time_derivative_stencil(time_indexing::Cyclical, times, n)
    Nt = length(times)
    n₋ = mod1(n - 1, Nt)
    n₊ = mod1(n + 1, Nt)
    Δt = @inbounds times[n₊] - times[n₋]
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
function ∂t(fts::FieldTimeSeriesLike)
    LX, LY, LZ = location(fts)
    times = fts.times
    time_indexing = fts.time_indexing

    Nt = length(times)
    minimum_length = time_indexing isa Cyclical ? 3 : 2
    Nt < minimum_length &&
        throw(ArgumentError("∂t requires at least $minimum_length times, got $Nt."))

    time_indexing isa Cyclical && isnothing(time_indexing.period) &&
        throw(ArgumentError("∂t with Cyclical time indexing requires an explicitly-specified period."))

    for source in extract_field_time_series(fts)
        window = time_indices_length(source.backend, source.times)
        if !isnothing(window) && window < 4
            throw(ArgumentError("∂t of a partly-in-memory FieldTimeSeries requires a window of at" *
                                " least 4 time indices: Time-interpolating the derivative touches" *
                                " one neighboring node on each side of the bracketing nodes."))
        end
    end

    args = (fts,)
    grid = fts.grid
    T = eltype(fts)

    interpolators = (nothing,) # the argument is colocated with its time derivative

    return FieldTimeSeriesOperation{LX, LY, LZ, typeof(time_indexing), TimeSeriesTimeDerivative,
                                    typeof(args), typeof(interpolators), typeof(grid), typeof(times), T}(
                                    TimeSeriesTimeDerivative(), args, interpolators, grid, times, time_indexing)
end

# The slice at node n couples the two neighboring nodes rather than same-index slices.
function Base.getindex(fts_op::TimeDerivativeFTSOp, n::Int)
    a = first(fts_op.args)
    n₋, n₊, Δt = time_derivative_stencil(fts_op.time_indexing, fts_op.times, n)
    update_field_time_series!(a, n₋, n₊)
    return (a[n₊] - a[n₋]) / Δt
end

# Updating the derivative at (n₁, n₂) requires the argument one node beyond each side.
function update_field_time_series!(fts_op::TimeDerivativeFTSOp, n₁::Int, n₂=n₁)
    a = first(fts_op.args)
    lo, _ = time_derivative_stencil(fts_op.time_indexing, fts_op.times, n₁)
    _, hi = time_derivative_stencil(fts_op.time_indexing, fts_op.times, n₂)
    return update_field_time_series!(a, lo, hi)
end

#####
##### Operators: the same operations that build AbstractOperations from Fields
##### build FieldTimeSeriesOperations from FieldTimeSeries. The per-arity method
##### definers below are called by the @unary, @binary, and @multiary macros
##### (via the Oceananigans.AbstractOperations.add_time_series_methods! hook),
##### so user-registered operators get FieldTimeSeries methods automatically.
#####

function Oceananigans.AbstractOperations.add_time_series_methods!(op, ::Val{1})
    @eval (::typeof($op))(a::FieldTimeSeriesLike) = FieldTimeSeriesOperation($op, a)
    return nothing
end

function Oceananigans.AbstractOperations.add_time_series_methods!(op, ::Val{2})
    @eval begin
        (::typeof($op))(a::FieldTimeSeriesLike, b::FieldTimeSeriesLike) = FieldTimeSeriesOperation($op, a, b)
        (::typeof($op))(a::FieldTimeSeriesLike, b::AbstractField) = FieldTimeSeriesOperation($op, a, b)
        (::typeof($op))(a::AbstractField, b::FieldTimeSeriesLike) = FieldTimeSeriesOperation($op, a, b)
        (::typeof($op))(a::FieldTimeSeriesLike, b::Number) = FieldTimeSeriesOperation($op, a, b)
        (::typeof($op))(a::Number, b::FieldTimeSeriesLike) = FieldTimeSeriesOperation($op, a, b)
    end
    return nothing
end

function Oceananigans.AbstractOperations.add_time_series_methods!(op, ::Val{3})
    @eval (::typeof($op))(a::FieldTimeSeriesLike, b::FieldTimeSeriesLike, c::FieldTimeSeriesLike, d::FieldTimeSeriesLike...) =
        FieldTimeSeriesOperation($op, a, b, c, d...)
    return nothing
end

# FieldTimeSeries methods for the default operators, whose @unary/@binary/@multiary
# registrations expand while AbstractOperations loads, before this module exists.
# Keep in sync with the operator lists at the bottom of
# src/AbstractOperations/AbstractOperations.jl.
for op in (sqrt, sin, cos, exp, tanh, abs, log10, log, tan, sinh, cosh, -, +)
    Oceananigans.AbstractOperations.add_time_series_methods!(op, Val(1))
end

for op in (+, -, *, /, ^, >, <, >=, <=, atan, atand, mod)
    Oceananigans.AbstractOperations.add_time_series_methods!(op, Val(2))
end

for op in (+, *)
    Oceananigans.AbstractOperations.add_time_series_methods!(op, Val(3))
end

# Spatial derivatives: the slice at time index n is the Derivative operation over the
# sliced argument, interpolation and location flipping included.
for op in (∂x, ∂y, ∂z)
    Oceananigans.AbstractOperations.add_time_series_methods!(op, Val(1))
end

#####
##### GPU adaptation: pointwise, slice-free evaluation inside kernels
#####

struct GPUAdaptedFieldTimeSeriesOperation{LX, LY, LZ, TI, O, A, P, G, χ, T} <: AbstractField{LX, LY, LZ, Nothing, T, 4}
    op :: O
    args :: A
    interpolators :: P
    grid :: G
    times :: χ
    time_indexing :: TI
end

# A lazy time slice of a four-dimensional series, indexable at (i, j, k) inside kernels.
struct TimeSlice{S, N}
    series :: S
    n :: N
end

@inline Base.getindex(slice::TimeSlice, i, j, k) = @inbounds slice.series[i, j, k, slice.n]

@inline time_series_value(a::Union{GPUAdaptedFieldTimeSeries, GPUAdaptedFieldTimeSeriesOperation}, i, j, k, n) =
    @inbounds a[i, j, k, n]

@inline time_series_value(a::AbstractArray, i, j, k, n) = @inbounds a[i, j, k]
@inline time_series_value(a, i, j, k, n) = a

@inline time_slice(a::Union{GPUAdaptedFieldTimeSeries, GPUAdaptedFieldTimeSeriesOperation}, n) = TimeSlice(a, n)
@inline time_slice(a, n) = a

# Pointwise node evaluation: spatially interpolate each argument (lazily time-sliced
# for series arguments) to the operation's location, then apply the operator — proper
# four-dimensional indexing with no intermediate slice or Field construction.
@inline Base.getindex(fts_op::GPUAdaptedFieldTimeSeriesOperation, i::Int, j::Int, k::Int, n::Int) =
    fts_op.op(map((▶, a) -> interpolated_argument_value(▶, a, i, j, k, n, fts_op.grid),
                  fts_op.interpolators, fts_op.args)...)

const SomeTimeSeries = Union{AbstractFieldTimeSeries, GPUAdaptedFieldTimeSeries, GPUAdaptedFieldTimeSeriesOperation}

@inline interpolated_argument_value(::Nothing, a, i, j, k, n, grid) = a
@inline interpolated_argument_value(▶, a::SomeTimeSeries, i, j, k, n, grid) = ▶(i, j, k, grid, TimeSlice(a, n))
@inline interpolated_argument_value(▶, a::AbstractArray, i, j, k, n, grid) = ▶(i, j, k, grid, a)

# Kernel-function form: the kernel function indexes its (time-sliced) arguments itself,
# so arguments at different locations are its responsibility, as for any
# KernelFunctionOperation.
@inline function Base.getindex(fts_op::GPUAdaptedFieldTimeSeriesOperation{<:Any, <:Any, <:Any, <:Any, <:TimeSeriesKernelFunction},
                               i::Int, j::Int, k::Int, n::Int)
    kf = fts_op.op
    return kf.func(i, j, k, kf.grid, map(a -> time_slice(a, n), fts_op.args)...)
end

@inline Base.getindex(fts_op::GPUAdaptedFieldTimeSeriesOperation{<:Any, <:Any, <:Any, <:Any, <:Union{typeof(∂x), typeof(∂y), typeof(∂z)}},
                      i::Int, j::Int, k::Int, n::Int) =
    first(fts_op.interpolators)(i, j, k, fts_op.grid, TimeSlice(first(fts_op.args), n))

@inline Base.getindex(fts_op::GPUAdaptedFieldTimeSeriesOperation, i::Int, j::Int, k::Int, time_index::Time) =
    interpolating_getindex(fts_op, i, j, k, time_index)

function Adapt.adapt_structure(to, fts_op::FieldTimeSeriesOperation{LX, LY, LZ}) where {LX, LY, LZ}
    op = Adapt.adapt(to, fts_op.op)
    args = map(a -> Adapt.adapt(to, a), fts_op.args)
    interpolators = Adapt.adapt(to, fts_op.interpolators)
    grid = Adapt.adapt(to, fts_op.grid)
    times = Adapt.adapt(to, fts_op.times)
    time_indexing = Adapt.adapt(to, fts_op.time_indexing)
    T = eltype(fts_op)

    return GPUAdaptedFieldTimeSeriesOperation{LX, LY, LZ, typeof(time_indexing), typeof(op), typeof(args),
                                              typeof(interpolators), typeof(grid), typeof(times), T}(
                                              op, args, interpolators, grid, times, time_indexing)
end

function Adapt.adapt_structure(to, kf::TimeSeriesKernelFunction{LX, LY, LZ}) where {LX, LY, LZ}
    func = Adapt.adapt(to, kf.func)
    grid = Adapt.adapt(to, kf.grid)
    return TimeSeriesKernelFunction{LX, LY, LZ, typeof(func), typeof(grid)}(func, grid)
end

on_architecture(to, fts_op::FieldTimeSeriesOperation) =
    FieldTimeSeriesOperation(on_architecture(to, fts_op.op),
                             map(a -> on_architecture(to, a), fts_op.args)...)

function on_architecture(to, kf::TimeSeriesKernelFunction{LX, LY, LZ}) where {LX, LY, LZ}
    grid = on_architecture(to, kf.grid)
    return TimeSeriesKernelFunction{LX, LY, LZ, typeof(kf.func), typeof(grid)}(kf.func, grid)
end

#####
##### Pointwise (slice-free) evaluation
#####

# Building the three-dimensional slice operation on every pointwise access is expensive
# (glwagner's review suggestion on #5761). Pointwise four-dimensional indexing spatially
# interpolates each argument with the operators stored at construction, so the only
# remaining requirement is that no window updates can be triggered by access — every
# series argument totally in memory. This depends only on argument types, so getindex's
# branch constant-folds.
@inline pointwise_evaluable(fts_op::FieldTimeSeriesOperation) =
    all(map(pointwise_evaluable_argument, fts_op.args))

@inline pointwise_evaluable_argument(a) = true
@inline pointwise_evaluable_argument(fts::FieldTimeSeries) = fts.backend isa TotallyInMemory
@inline pointwise_evaluable_argument(fts_op::FieldTimeSeriesOperation) = pointwise_evaluable(fts_op)

@inline time_series_value(a::FieldTimeSeries, i, j, k, n) = @inbounds a[i, j, k, n]
@inline time_series_value(fts_op::FieldTimeSeriesOperation, i, j, k, n) = @inbounds fts_op[i, j, k, n]
@inline time_slice(a::FieldTimeSeriesLike, n) = TimeSlice(a, n)

@propagate_inbounds pointwise_getindex(fts_op::FieldTimeSeriesOperation, i, j, k, n) =
    fts_op.op(map((▶, a) -> interpolated_argument_value(▶, a, i, j, k, n, fts_op.grid),
                  fts_op.interpolators, fts_op.args)...)

# Spatial derivatives evaluate their stencil-and-interpolation pair directly: the
# operator is the stencil, so it must not be re-applied to the resulting value.
const SpatialDerivative = Union{typeof(∂x), typeof(∂y), typeof(∂z)}
const SpatialDerivativeFTSOp = FieldTimeSeriesOperation{<:Any, <:Any, <:Any, <:Any, <:SpatialDerivative}

@propagate_inbounds pointwise_getindex(fts_op::SpatialDerivativeFTSOp, i, j, k, n) =
    first(fts_op.interpolators)(i, j, k, fts_op.grid, TimeSlice(first(fts_op.args), n))

@propagate_inbounds function pointwise_getindex(fts_op::FieldTimeSeriesOperation{<:Any, <:Any, <:Any, <:Any, <:TimeSeriesKernelFunction},
                                                i, j, k, n)
    kf = fts_op.op
    return kf.func(i, j, k, kf.grid, map(a -> time_slice(a, n), fts_op.args)...)
end

@propagate_inbounds function pointwise_getindex(fts_op::TimeDerivativeFTSOp, i, j, k, n)
    a = first(fts_op.args)
    n₋, n₊, Δt = time_derivative_stencil(fts_op.time_indexing, fts_op.times, n)
    return (time_series_value(a, i, j, k, n₊) - time_series_value(a, i, j, k, n₋)) / Δt
end

const TimeDerivativeGPUFTSOp = GPUAdaptedFieldTimeSeriesOperation{<:Any, <:Any, <:Any, <:Any, <:TimeSeriesTimeDerivative}

@inline function Base.getindex(fts_op::TimeDerivativeGPUFTSOp, i::Int, j::Int, k::Int, n::Int)
    a = first(fts_op.args)
    n₋, n₊, Δt = time_derivative_stencil(fts_op.time_indexing, fts_op.times, n)
    return (time_series_value(a, i, j, k, n₊) - time_series_value(a, i, j, k, n₋)) / Δt
end

#####
##### Reductions
#####

# Lazy Average and Integral of a series: the slice at time index n is the Scan over the
# sliced argument. Scans require a full sweep, so these operations are never pointwise
# evaluable; index them via slices (`Field(op[n])`) or materialize them.
struct TimeSeriesReduction{R, K}
    scan :: R
    kwargs :: K
end

(tsr::TimeSeriesReduction)(a) = tsr.scan(a; tsr.kwargs...)

Base.show(io::IO, tsr::TimeSeriesReduction) = print(io, tsr.scan, " with ", tsr.kwargs)

const TimeSeriesReductionFTSOp = FieldTimeSeriesOperation{<:Any, <:Any, <:Any, <:Any, <:TimeSeriesReduction}

@inline pointwise_evaluable(::TimeSeriesReductionFTSOp) = false

# Scans cannot be broadcast onto a Field, so materialize reduced slices via compute!.
function set!(fts::InMemoryFTS, fts_op::TimeSeriesReductionFTSOp; kwargs...)
    for n in time_indices(fts)
        set!(fts[n], compute!(Field(fts_op[n])))
    end
    return fts
end

Oceananigans.AbstractOperations.Average(fts::FieldTimeSeriesLike; kwargs...) =
    FieldTimeSeriesOperation(TimeSeriesReduction(Average, NamedTuple(kwargs)), fts)

Oceananigans.AbstractOperations.Integral(fts::FieldTimeSeriesLike; kwargs...) =
    FieldTimeSeriesOperation(TimeSeriesReduction(Integral, NamedTuple(kwargs)), fts)

# Reductions over FieldTimeSeriesOperation mirror the FieldTimeSeries methods in
# field_time_series_reductions.jl: spatial dims produce a reduced FieldTimeSeries,
# dims including 4 reduce over time, and Colon reduces over everything. Slices are
# reduced lazily — reductions apply pointwise to operations without materializing them.
for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)
    reduction! = Symbol(reduction, '!')

    @eval begin
        function Base.$(reduction)(f::Function, fts_op::FieldTimeSeriesOperation; dims=:, kw...)
            dims = tupleit(dims)
            Nt = length(fts_op.times)

            if dims isa Colon
                return Base.$(reduction)(Base.$(reduction)(f, fts_op[n]; kw...) for n in 1:Nt)
            elseif 4 ∈ dims
                spatial_dims = filter(d -> d != 4, dims)
                loc = isempty(spatial_dims) ? instantiated_location(fts_op) :
                                              reduced_location(instantiated_location(fts_op); dims=spatial_dims)
                rts = FieldTimeSeries(loc, fts_op.grid, [mean(fts_op.times)]; indices=indices(fts_op))

                if isempty(spatial_dims)
                    temp = compute!(Field(fts_op[1]))
                    set!(rts[1], temp)
                    for n in 2:Nt
                        set!(temp, fts_op[n])
                        broadcasted_accumulate!(Base.$(reduction!), interior(rts[1]), interior(temp))
                    end
                else
                    temp = Base.$(reduction)(f, fts_op[1]; dims=spatial_dims, kw...)
                    set!(rts[1], temp)
                    for n in 2:Nt
                        Base.$(reduction!)(f, temp, fts_op[n]; kw...)
                        broadcasted_accumulate!(Base.$(reduction!), interior(rts[1]), interior(temp))
                    end
                end
            else
                loc = reduced_location(instantiated_location(fts_op); dims)
                rts = FieldTimeSeries(loc, fts_op.grid, fts_op.times; indices=indices(fts_op))
                for n in 1:Nt
                    Base.$(reduction!)(f, rts[n], fts_op[n]; kw...)
                end
            end

            return rts
        end

        Base.$(reduction)(fts_op::FieldTimeSeriesOperation; kw...) = Base.$(reduction)(identity, fts_op; kw...)
    end
end

Oceananigans.Fields.conditional_length(fts_op::FieldTimeSeriesOperation) =
    length(fts_op.times) * Oceananigans.Fields.conditional_length(fts_op[1])

function Statistics._mean(f, fts_op::FieldTimeSeriesOperation, ::Colon; condition=nothing)
    isnothing(condition) || error("Conditional mean of a FieldTimeSeriesOperation is not implemented.")
    return sum(f, fts_op) / Oceananigans.Fields.conditional_length(fts_op)
end

function Statistics._mean(f, fts_op::FieldTimeSeriesOperation, dims; condition=nothing)
    isnothing(condition) || error("Conditional mean of a FieldTimeSeriesOperation is not implemented.")
    r = sum(f, fts_op; dims)
    n = Oceananigans.Fields.conditional_length(fts_op, dims)
    r ./= n
    return r
end
