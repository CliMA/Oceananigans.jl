using Oceananigans.Fields: AbstractField, instantiated_location, compute_at!, indices, filter_nothing_dims
using Oceananigans.Grids: RectilinearGrid, Face, Center, Bounded, Flat, topology, interior_indices
using Oceananigans.Grids: ξnodes, ηnodes, rnodes
using Oceananigans.Architectures: architecture, on_architecture, CPU
using Oceananigans.Utils: KernelParameters, launch!, tupleit
using Oceananigans.Grids: inactive_node

using KernelAbstractions: @kernel, @index, @atomic

const HISTOGRAM_MEMORY_WARNING_BYTES = 512 * 1024^2 # 0.5 GiB across local + global buffers

abstract type AbstractHistogramWeights end

struct HistogramCountWeights <: AbstractHistogramWeights end

struct HistogramFieldWeights{W} <: AbstractHistogramWeights
    field :: W
end

struct HistogramIntegralCountWeights{M} <: AbstractHistogramWeights
    metric :: M
end

struct HistogramIntegralFieldWeights{W, M} <: AbstractHistogramWeights
    field :: W
    metric :: M
end

struct HistogramAverageCountWeights{M} <: AbstractHistogramWeights
    metric :: M
end

struct HistogramAverageFieldWeights{W, M} <: AbstractHistogramWeights
    field :: W
    metric :: M
end

Adapt.adapt_structure(to, ::HistogramCountWeights) = HistogramCountWeights()
Adapt.adapt_structure(to, weights::HistogramFieldWeights) = HistogramFieldWeights(adapt(to, weights.field))
Adapt.adapt_structure(to, weights::HistogramIntegralCountWeights) = HistogramIntegralCountWeights(adapt(to, weights.metric))
Adapt.adapt_structure(to, weights::HistogramIntegralFieldWeights) = HistogramIntegralFieldWeights(adapt(to, weights.field), adapt(to, weights.metric))
Adapt.adapt_structure(to, weights::HistogramAverageCountWeights) = HistogramAverageCountWeights(adapt(to, weights.metric))
Adapt.adapt_structure(to, weights::HistogramAverageFieldWeights) = HistogramAverageFieldWeights(adapt(to, weights.field), adapt(to, weights.metric))

struct HistogramOperation{G, T, A, B, BN, E1, E2, W, H, LN, C, GN, K} <: AbstractOperation{Center, Center, Center, G, T}
    grid :: G
    a :: A
    b :: B
    bins :: BN
    edges1 :: E1
    edges2 :: E2
    weights :: W
    local_histogram :: H
    local_normalization :: LN
    global_cache :: C
    global_normalization :: GN
    launch_parameters :: K
    dims :: NTuple{N, Int} where N
    method :: Symbol
    reduced_dimensions :: NTuple{3, Bool}
    retained_offsets :: NTuple{3, Int}
    retained_lengths :: NTuple{3, Int}
end

function HistogramOperation(grid::G, a::A, b::B, bins::BN, edges1::E1, edges2::E2,
                            weights::W, local_histogram::H, local_normalization::LN,
                            global_cache::C, global_normalization::GN,
                            launch_parameters::K, dims::NTuple{N, Int}, method::Symbol,
                            reduced_dimensions::NTuple{3, Bool},
                            retained_offsets::NTuple{3, Int},
                            retained_lengths::NTuple{3, Int}) where {G, A, B, BN, E1, E2, W, H, LN, C, GN, K, N}
    T = eltype(global_cache)
    return HistogramOperation{G, T, A, B, BN, E1, E2, W, H, LN, C, GN, K}(grid, a, b, bins, edges1, edges2,
                                                                   weights, local_histogram, local_normalization,
                                                                   global_cache, global_normalization,
                                                                   launch_parameters, dims, method,
                                                                   reduced_dimensions, retained_offsets,
                                                                   retained_lengths)
end

struct Histogram1DOperation{G, T, A, BN, E1, W, H, LN, C, GN, K} <: AbstractOperation{Center, Center, Center, G, T}
    grid :: G
    a :: A
    bins :: BN
    edges1 :: E1
    weights :: W
    local_histogram :: H
    local_normalization :: LN
    global_cache :: C
    global_normalization :: GN
    launch_parameters :: K
    dims :: NTuple{N, Int} where N
    method :: Symbol
    reduced_dimensions :: NTuple{3, Bool}
    retained_offsets :: NTuple{3, Int}
    retained_lengths :: NTuple{3, Int}
end

function Histogram1DOperation(grid::G, a::A, bins::BN, edges1::E1,
                              weights::W, local_histogram::H, local_normalization::LN,
                              global_cache::C, global_normalization::GN,
                              launch_parameters::K, dims::NTuple{N, Int}, method::Symbol,
                              reduced_dimensions::NTuple{3, Bool},
                              retained_offsets::NTuple{3, Int},
                              retained_lengths::NTuple{3, Int}) where {G, A, BN, E1, W, H, LN, C, GN, K, N}
    T = eltype(global_cache)
    return Histogram1DOperation{G, T, A, BN, E1, W, H, LN, C, GN, K}(grid, a, bins, edges1,
                                                              weights, local_histogram, local_normalization,
                                                              global_cache, global_normalization,
                                                              launch_parameters, dims, method,
                                                              reduced_dimensions, retained_offsets,
                                                              retained_lengths)
end

@inline retained_dimension_face_nodes(grid, retained_dimension, face_indices) =
    retained_dimension === 1 ? ξnodes(grid, Face(); indices=face_indices) :
    retained_dimension === 2 ? ηnodes(grid, Face(); indices=face_indices) :
                               rnodes(grid, Face(); indices=face_indices)

function histogram_retained_coordinate(a, launch_indices, reduced_dimensions, retained_count, ::Type{FT}) where FT
    retained_dimensions = Tuple(d for d in 1:3 if !reduced_dimensions[d])

    if length(retained_dimensions) != 1
        return (zero(FT), FT(retained_count))
    end

    retained_dimension = only(retained_dimensions)
    retained_location = instantiated_location(a)[retained_dimension]

    if !(retained_location isa Center) || topology(a.grid, retained_dimension) !== Bounded
        return (zero(FT), FT(retained_count))
    end

    retained_indices = launch_indices[retained_dimension]
    face_indices = first(retained_indices):(last(retained_indices) + 1)
    retained_faces = retained_dimension_face_nodes(a.grid, retained_dimension, face_indices)
    retained_faces_cpu = collect(on_architecture(CPU(), retained_faces))

    return convert(Vector{FT}, retained_faces_cpu)
end

"""
    Histogram(a::AbstractField, b::AbstractField; bins=NamedTuple(), weights=:count, dims=(1, 2, 3), method=:sum)
    Histogram(fields::NamedTuple; bins=NamedTuple(), weights=:count, dims=(1, 2, 3), method=:sum)
    Histogram(a::AbstractField; bins, weights=:count, dims=(1, 2, 3), method=:sum)

Construct a weighted histogram operation from field operand(s).

The result is an `AbstractOperation` that can be used with `Field(...)` and output writers,
for example `Field(Histogram(...))`.

Constraints:
- `bins` must be strictly-increasing edge vectors:
  - 2D histogram: exactly two edge vectors.
  - 1D histogram: exactly one edge vector.
- `weights` must be:
  - with `method=:sum`: `:count` or an `AbstractField`;
  - with `method=:integral`: `:count` or an `AbstractField`.
  - with `method=:average`: `:count` or an `AbstractField`.
- `:count` with `method=:sum` accumulates one count per sample.
- `:count` with `method=:integral` accumulates the geometric integration metric for `dims`
  (`Δx`, `Δy`, `Δz`, `Az`, `Ay`, `Ax`, or `volume`), matching `Integral` semantics.
- `:average` computes a metric-weighted mean:
  `Histogram(...; weights=w, method=:integral) / Histogram(...; weights=:count, method=:integral)`
  with zero returned for bins whose denominator is zero.
- Typical use: `weights=tracer_field, method=:average` to compute a binned mean tracer.
- `weights=:count, method=:average` yields one for populated bins and zero for empty bins.
- `dims` is the subset of spatial dimensions to reduce over. `dims=:` is shorthand for `(1, 2, 3)`.
- `method` must be `:sum`, `:integral`, or `:average`.
- Distributed architectures are not supported yet.

For partial reductions, retained dimensions are flattened into histogram output indices.
"""
function Histogram(a::AbstractField, b::AbstractField;
                   bins = NamedTuple(),
                   weights = :count,
                   dims = (1, 2, 3),
                   method = :sum)

    validate_histogram_operands(a, b)

    dims = validate_histogram_dims(dims)
    dims = filter_nothing_dims(dims, instantiated_location(a))
    isempty(dims) &&
        throw(ArgumentError("Histogram dims cannot be empty after filtering dimensions absent from operand location $(instantiated_location(a))."))
    method = validate_histogram_method(method)
    bins = validate_histogram_bins_2d(bins)
    weights = validate_histogram_weights(weights, method, dims, a, b)

    arch = architecture(a.grid)
    validate_histogram_architecture(arch)

    FT = histogram_eltype(a, b, bins, weights)
    bins = convert_histogram_bins_eltype(bins, FT)
    edges1_cpu, edges2_cpu = values(bins)

    nbin1 = length(edges1_cpu) - 1
    nbin2 = length(edges2_cpu) - 1

    launch_indices = histogram_launch_indices(a, b, weights)
    launch_parameters = KernelParameters(launch_indices...)

    reduced_dimensions, retained_offsets, retained_lengths, retained_count =
        histogram_reduction_metadata(launch_indices, dims)
    retained_coordinate = histogram_retained_coordinate(a, launch_indices, reduced_dimensions, retained_count, FT)

    buffers = method === :average ? 4 : 2
    maybe_warn_histogram_memory((nbin1, nbin2, retained_count), FT; dims, buffers)

    histogram_grid = RectilinearGrid(CPU(), FT;
                                     size = (nbin1, nbin2, retained_count),
                                     topology = (Bounded, Bounded, Bounded),
                                     halo = (1, 1, 1),
                                     x = edges1_cpu,
                                     y = edges2_cpu,
                                     z = retained_coordinate)

    edges1 = on_architecture(arch, edges1_cpu)
    edges2 = on_architecture(arch, edges2_cpu)

    local_histogram = zeros(arch, FT, nbin1, nbin2, retained_count)
    global_cache = zeros(FT, nbin1, nbin2, retained_count)
    local_normalization = method === :average ? zeros(arch, FT, nbin1, nbin2, retained_count) : nothing
    global_normalization = method === :average ? zeros(FT, nbin1, nbin2, retained_count) : nothing

    return HistogramOperation(histogram_grid, a, b, bins, edges1, edges2, weights,
                              local_histogram, local_normalization,
                              global_cache, global_normalization, launch_parameters,
                              dims, method, reduced_dimensions,
                              retained_offsets, retained_lengths)
end

function Histogram(a::AbstractField;
                   bins = NamedTuple(),
                   weights = :count,
                   dims = (1, 2, 3),
                   method = :sum)

    dims = validate_histogram_dims(dims)
    dims = filter_nothing_dims(dims, instantiated_location(a))
    isempty(dims) &&
        throw(ArgumentError("Histogram dims cannot be empty after filtering dimensions absent from operand location $(instantiated_location(a))."))
    method = validate_histogram_method(method)
    bins = validate_histogram_bins_1d(bins)
    weights = validate_histogram_weights(weights, method, dims, a)

    arch = architecture(a.grid)
    validate_histogram_architecture(arch)

    FT = histogram_eltype(a, bins, weights)
    bins = convert_histogram_bins_eltype(bins, FT)
    edges1_cpu = first(values(bins))

    nbin1 = length(edges1_cpu) - 1

    launch_indices = histogram_launch_indices(a, weights)
    launch_parameters = KernelParameters(launch_indices...)

    reduced_dimensions, retained_offsets, retained_lengths, retained_count =
        histogram_reduction_metadata(launch_indices, dims)
    retained_coordinate = histogram_retained_coordinate(a, launch_indices, reduced_dimensions, retained_count, FT)

    buffers = method === :average ? 4 : 2
    maybe_warn_histogram_memory((nbin1, retained_count, 1), FT; dims, buffers)

    histogram_grid = RectilinearGrid(CPU(), FT;
                                     size = (nbin1, retained_count),
                                     topology = (Bounded, Bounded, Flat),
                                     halo = (1, 1),
                                     x = edges1_cpu,
                                     y = retained_coordinate)

    edges1 = on_architecture(arch, edges1_cpu)

    local_histogram = zeros(arch, FT, nbin1, retained_count, 1)
    global_cache = zeros(FT, nbin1, retained_count, 1)
    local_normalization = method === :average ? zeros(arch, FT, nbin1, retained_count, 1) : nothing
    global_normalization = method === :average ? zeros(FT, nbin1, retained_count, 1) : nothing

    return Histogram1DOperation(histogram_grid, a, bins, edges1, weights,
                                local_histogram, local_normalization,
                                global_cache, global_normalization, launch_parameters,
                                dims, method, reduced_dimensions,
                                retained_offsets, retained_lengths)
end

function Histogram(fields::NamedTuple;
                   bins = NamedTuple(),
                   weights = :count,
                   dims = (1, 2, 3),
                   method = :sum)

    if length(fields) == 2
        operand_names = keys(fields)
        a, b = values(fields)

        a isa AbstractField ||
            throw(ArgumentError("Histogram(fields=...) requires field operands, but `$(operand_names[1])` is $(typeof(a))."))

        b isa AbstractField ||
            throw(ArgumentError("Histogram(fields=...) requires field operands, but `$(operand_names[2])` is $(typeof(b))."))

        bins = reorder_histogram_bins_for_named_operands(bins, operand_names)
        return Histogram(a, b; bins, weights, dims, method)
    elseif length(fields) == 1
        operand_name = first(keys(fields))
        a = first(values(fields))

        a isa AbstractField ||
            throw(ArgumentError("Histogram(fields=...) requires field operands, but `$(operand_name)` is $(typeof(a))."))

        bins = bins isa NamedTuple ? reorder_histogram_bins_for_named_operand(bins, operand_name) : bins
        return Histogram(a; bins, weights, dims, method)
    else
        throw(ArgumentError("Histogram(fields=...) requires exactly one or two named field operands."))
    end
end

Base.summary(::HistogramCountWeights) = ":count"
Base.summary(w::HistogramFieldWeights) = "field ($(summary(w.field)))"
Base.summary(w::HistogramIntegralCountWeights) = "integral(:count)"
Base.summary(w::HistogramIntegralFieldWeights) = "integral(field ($(summary(w.field))))"
Base.summary(w::HistogramAverageCountWeights) = "average(:count)"
Base.summary(w::HistogramAverageFieldWeights) = "average(field ($(summary(w.field))))"

function Base.summary(op::HistogramOperation)
    nbin1, nbin2, nret = size(op.global_cache)
    return "Histogram2D ($nbin1 × $nbin2 bins, retained=$nret)"
end

function Base.summary(op::Histogram1DOperation)
    nbin1, nret, _ = size(op.global_cache)
    return "Histogram1D ($nbin1 bins, retained=$nret)"
end

function Base.show(io::IO, op::HistogramOperation)
    print(io, summary(op), '\n',
              "├── method: ", op.method, '\n',
              "├── dims: ", op.dims, '\n',
              "├── bins: ", keys(op.bins), '\n',
              "├── weights: ", summary(op.weights), '\n',
              "├── operand 1: ", summary(op.a), '\n',
              "└── operand 2: ", summary(op.b))
end

function Base.show(io::IO, op::Histogram1DOperation)
    print(io, summary(op), '\n',
              "├── method: ", op.method, '\n',
              "├── dims: ", op.dims, '\n',
              "├── bins: ", keys(op.bins), '\n',
              "├── weights: ", summary(op.weights), '\n',
              "└── operand: ", summary(op.a))
end

@inline Base.getindex(op::HistogramOperation, i, j, k) = @inbounds op.global_cache[i, j, k]
@inline Base.getindex(op::Histogram1DOperation, i, j, k) = @inbounds op.global_cache[i, j, k]

function compute_at!(op::HistogramOperation, time)
    compute_at!(op.a, time)
    compute_at!(op.b, time)
    compute_histogram_weights_at!(op.weights, time)

    fill!(op.local_histogram, zero(eltype(op.local_histogram)))

    grid = op.a.grid
    arch = architecture(grid)

    if op.method === :average
        fill!(op.local_normalization, zero(eltype(op.local_normalization)))

        launch!(arch, grid, op.launch_parameters, _accumulate_histogram_2d_average!,
                op.local_histogram, op.local_normalization, op.a, op.b, op.edges1, op.edges2,
                grid, op.weights, instantiated_location(op.a),
                op.reduced_dimensions, op.retained_offsets, op.retained_lengths)

        copyto!(op.global_cache, on_architecture(CPU(), op.local_histogram))
        copyto!(op.global_normalization, on_architecture(CPU(), op.local_normalization))
        finalize_average_histogram!(op.global_cache, op.global_normalization)
    else
        launch!(arch, grid, op.launch_parameters, _accumulate_histogram_2d!,
                op.local_histogram, op.a, op.b, op.edges1, op.edges2,
                grid, op.weights, instantiated_location(op.a),
                op.reduced_dimensions, op.retained_offsets, op.retained_lengths)

        local_histogram_cpu = on_architecture(CPU(), op.local_histogram)
        copyto!(op.global_cache, local_histogram_cpu)
    end

    return nothing
end

function compute_at!(op::Histogram1DOperation, time)
    compute_at!(op.a, time)
    compute_histogram_weights_at!(op.weights, time)

    fill!(op.local_histogram, zero(eltype(op.local_histogram)))

    grid = op.a.grid
    arch = architecture(grid)

    if op.method === :average
        fill!(op.local_normalization, zero(eltype(op.local_normalization)))

        launch!(arch, grid, op.launch_parameters, _accumulate_histogram_1d_average!,
                op.local_histogram, op.local_normalization, op.a, op.edges1,
                grid, op.weights, instantiated_location(op.a),
                op.reduced_dimensions, op.retained_offsets, op.retained_lengths)

        copyto!(op.global_cache, on_architecture(CPU(), op.local_histogram))
        copyto!(op.global_normalization, on_architecture(CPU(), op.local_normalization))
        finalize_average_histogram!(op.global_cache, op.global_normalization)
    else
        launch!(arch, grid, op.launch_parameters, _accumulate_histogram_1d!,
                op.local_histogram, op.a, op.edges1,
                grid, op.weights, instantiated_location(op.a),
                op.reduced_dimensions, op.retained_offsets, op.retained_lengths)

        local_histogram_cpu = on_architecture(CPU(), op.local_histogram)
        copyto!(op.global_cache, local_histogram_cpu)
    end

    return nothing
end

@inline function finalize_average_histogram!(numerator, denominator)
    @. numerator = ifelse(denominator > 0, numerator / denominator, zero(eltype(numerator)))
    return nothing
end

@inline compute_histogram_weights_at!(::HistogramCountWeights, time) = nothing
@inline compute_histogram_weights_at!(weights::HistogramFieldWeights, time) = compute_at!(weights.field, time)
@inline compute_histogram_weights_at!(weights::HistogramIntegralCountWeights, time) = compute_at!(weights.metric, time)
@inline function compute_histogram_weights_at!(weights::HistogramIntegralFieldWeights, time)
    compute_at!(weights.field, time)
    compute_at!(weights.metric, time)
    return nothing
end
@inline compute_histogram_weights_at!(weights::HistogramAverageCountWeights, time) = compute_at!(weights.metric, time)
@inline function compute_histogram_weights_at!(weights::HistogramAverageFieldWeights, time)
    compute_at!(weights.field, time)
    compute_at!(weights.metric, time)
    return nothing
end

@inline histogram_weight(i, j, k, grid, ::HistogramCountWeights) = 1
@inline histogram_weight(i, j, k, grid, weights::HistogramFieldWeights) = @inbounds weights.field[i, j, k]
@inline histogram_weight(i, j, k, grid, weights::HistogramIntegralCountWeights) = @inbounds weights.metric[i, j, k]
@inline histogram_weight(i, j, k, grid, weights::HistogramIntegralFieldWeights) = @inbounds weights.field[i, j, k] * weights.metric[i, j, k]
@inline histogram_weight(i, j, k, grid, weights::HistogramAverageCountWeights) = @inbounds weights.metric[i, j, k]
@inline histogram_weight(i, j, k, grid, weights::HistogramAverageFieldWeights) = @inbounds weights.field[i, j, k] * weights.metric[i, j, k]

@inline histogram_normalization_weight(i, j, k, grid, weights::HistogramAverageCountWeights) = @inbounds weights.metric[i, j, k]
@inline histogram_normalization_weight(i, j, k, grid, weights::HistogramAverageFieldWeights) = @inbounds weights.metric[i, j, k]

@inline function find_histogram_bin(value, edges)
    N = length(edges)

    @inbounds begin
        value < edges[1] && return 0
        value > edges[N] && return 0
        value == edges[N] && return N - 1
    end

    lo = 1
    hi = N

    while lo <= hi
        mid = (lo + hi) >>> 1
        @inbounds edge = edges[mid]

        if edge <= value
            lo = mid + 1
        else
            hi = mid - 1
        end
    end

    return ifelse(1 <= hi < N, hi, 0)
end

@inline function retained_linear_index(i, j, k,
                                       reduced_dimensions::NTuple{3, Bool},
                                       retained_offsets::NTuple{3, Int},
                                       retained_lengths::NTuple{3, Int})
    iR = reduced_dimensions[1] ? 1 : i - retained_offsets[1] + 1
    jR = reduced_dimensions[2] ? 1 : j - retained_offsets[2] + 1
    kR = reduced_dimensions[3] ? 1 : k - retained_offsets[3] + 1

    return iR + retained_lengths[1] * ((jR - 1) + retained_lengths[2] * (kR - 1))
end

@kernel function _accumulate_histogram_2d!(histogram, a, b, edges1, edges2,
                                           grid, weights, loc,
                                           reduced_dimensions,
                                           retained_offsets,
                                           retained_lengths)
    i, j, k = @index(Global, NTuple)

    if !inactive_node(i, j, k, grid, loc...)
        @inbounds aᵢ = a[i, j, k]
        @inbounds bᵢ = b[i, j, k]

        ibin1 = find_histogram_bin(aᵢ, edges1)
        ibin2 = find_histogram_bin(bᵢ, edges2)

        in_range = (ibin1 > 0) & (ibin2 > 0)
        if in_range
            retained_index = retained_linear_index(i, j, k, reduced_dimensions, retained_offsets, retained_lengths)
            weight = convert(eltype(histogram), histogram_weight(i, j, k, grid, weights))
            @atomic histogram[ibin1, ibin2, retained_index] += weight
        end
    end
end

@kernel function _accumulate_histogram_2d_average!(histogram, normalization, a, b, edges1, edges2,
                                                   grid, weights, loc,
                                                   reduced_dimensions,
                                                   retained_offsets,
                                                   retained_lengths)
    i, j, k = @index(Global, NTuple)

    if !inactive_node(i, j, k, grid, loc...)
        @inbounds aᵢ = a[i, j, k]
        @inbounds bᵢ = b[i, j, k]

        ibin1 = find_histogram_bin(aᵢ, edges1)
        ibin2 = find_histogram_bin(bᵢ, edges2)

        in_range = (ibin1 > 0) & (ibin2 > 0)
        if in_range
            retained_index = retained_linear_index(i, j, k, reduced_dimensions, retained_offsets, retained_lengths)
            numerator_weight = convert(eltype(histogram), histogram_weight(i, j, k, grid, weights))
            denominator_weight = convert(eltype(normalization), histogram_normalization_weight(i, j, k, grid, weights))
            @atomic histogram[ibin1, ibin2, retained_index] += numerator_weight
            @atomic normalization[ibin1, ibin2, retained_index] += denominator_weight
        end
    end
end

@kernel function _accumulate_histogram_1d!(histogram, a, edges1,
                                           grid, weights, loc,
                                           reduced_dimensions,
                                           retained_offsets,
                                           retained_lengths)
    i, j, k = @index(Global, NTuple)

    if !inactive_node(i, j, k, grid, loc...)
        @inbounds aᵢ = a[i, j, k]

        ibin1 = find_histogram_bin(aᵢ, edges1)

        if ibin1 > 0
            retained_index = retained_linear_index(i, j, k, reduced_dimensions, retained_offsets, retained_lengths)
            weight = convert(eltype(histogram), histogram_weight(i, j, k, grid, weights))
            @atomic histogram[ibin1, retained_index, 1] += weight
        end
    end
end

@kernel function _accumulate_histogram_1d_average!(histogram, normalization, a, edges1,
                                                   grid, weights, loc,
                                                   reduced_dimensions,
                                                   retained_offsets,
                                                   retained_lengths)
    i, j, k = @index(Global, NTuple)

    if !inactive_node(i, j, k, grid, loc...)
        @inbounds aᵢ = a[i, j, k]

        ibin1 = find_histogram_bin(aᵢ, edges1)

        if ibin1 > 0
            retained_index = retained_linear_index(i, j, k, reduced_dimensions, retained_offsets, retained_lengths)
            numerator_weight = convert(eltype(histogram), histogram_weight(i, j, k, grid, weights))
            denominator_weight = convert(eltype(normalization), histogram_normalization_weight(i, j, k, grid, weights))
            @atomic histogram[ibin1, retained_index, 1] += numerator_weight
            @atomic normalization[ibin1, retained_index, 1] += denominator_weight
        end
    end
end

@inline histogram_weight_operands(::HistogramCountWeights) = ()
@inline histogram_weight_operands(weights::HistogramFieldWeights) = (weights.field,)
@inline histogram_weight_operands(weights::HistogramIntegralCountWeights) = (weights.metric,)
@inline histogram_weight_operands(weights::HistogramIntegralFieldWeights) = (weights.field, weights.metric)
@inline histogram_weight_operands(weights::HistogramAverageCountWeights) = (weights.metric,)
@inline histogram_weight_operands(weights::HistogramAverageFieldWeights) = (weights.field, weights.metric)

function histogram_launch_indices(a, b, weights)
    loc = instantiated_location(a)
    grid = a.grid

    idx1 = restricted_field_indices(a, 1, loc, grid)
    idx2 = restricted_field_indices(a, 2, loc, grid)
    idx3 = restricted_field_indices(a, 3, loc, grid)

    idx1 = intersect_histogram_ranges(idx1, restricted_field_indices(b, 1, loc, grid))
    idx2 = intersect_histogram_ranges(idx2, restricted_field_indices(b, 2, loc, grid))
    idx3 = intersect_histogram_ranges(idx3, restricted_field_indices(b, 3, loc, grid))

    for weight_operand in histogram_weight_operands(weights)
        idx1 = intersect_histogram_ranges(idx1, restricted_field_indices(weight_operand, 1, loc, grid))
        idx2 = intersect_histogram_ranges(idx2, restricted_field_indices(weight_operand, 2, loc, grid))
        idx3 = intersect_histogram_ranges(idx3, restricted_field_indices(weight_operand, 3, loc, grid))
    end

    return (idx1, idx2, idx3)
end

function histogram_launch_indices(a, weights)
    loc = instantiated_location(a)
    grid = a.grid

    idx1 = restricted_field_indices(a, 1, loc, grid)
    idx2 = restricted_field_indices(a, 2, loc, grid)
    idx3 = restricted_field_indices(a, 3, loc, grid)

    for weight_operand in histogram_weight_operands(weights)
        idx1 = intersect_histogram_ranges(idx1, restricted_field_indices(weight_operand, 1, loc, grid))
        idx2 = intersect_histogram_ranges(idx2, restricted_field_indices(weight_operand, 2, loc, grid))
        idx3 = intersect_histogram_ranges(idx3, restricted_field_indices(weight_operand, 3, loc, grid))
    end

    return (idx1, idx2, idx3)
end

function histogram_reduction_metadata(launch_indices::NTuple{3, <:AbstractUnitRange}, dims)
    reduced_dimensions = (1 in dims, 2 in dims, 3 in dims)
    retained_offsets = ntuple(d -> first(launch_indices[d]), 3)
    retained_lengths = ntuple(d -> reduced_dimensions[d] ? 1 : length(launch_indices[d]), 3)
    retained_count = prod(retained_lengths)

    return reduced_dimensions, retained_offsets, retained_lengths, retained_count
end

function maybe_warn_histogram_memory(shape, ::Type{FT}; dims, buffers=2) where FT
    bytes = buffers * sizeof(FT) * prod(shape)

    if bytes > HISTOGRAM_MEMORY_WARNING_BYTES
        gib = round(bytes / 1024^3; digits=3)
        @warn "Histogram allocation is large (local + global buffers)." dims shape eltype=FT estimated_gib=gib
    end

    return nothing
end

function restricted_field_indices(field, dim, loc, grid)
    topo = topology(grid, dim)()
    interior = interior_indices(loc[dim], topo, size(grid, dim))
    return intersect_histogram_ranges(interior, indices(field)[dim])
end

intersect_histogram_ranges(interior::AbstractUnitRange, ::Colon) = interior

function intersect_histogram_ranges(i1::AbstractUnitRange, i2::AbstractUnitRange)
    i = max(first(i1), first(i2)):min(last(i1), last(i2))
    first(i) > last(i) && throw(ArgumentError("Histogram operands do not have intersecting indices."))
    return i
end

function validate_histogram_operands(a, b)
    a.grid == b.grid ||
        throw(ArgumentError("Histogram operands must be on the same grid."))

    instantiated_location(a) == instantiated_location(b) ||
        throw(ArgumentError("Histogram operands must have the same location. Histogram does not perform automatic co-location/interpolation."))

    return nothing
end

validate_histogram_dims(::Colon) = (1, 2, 3)

function validate_histogram_dims(dims)
    dims_tuple = Tuple(sort(unique(tupleit(dims))))

    isempty(dims_tuple) &&
        throw(ArgumentError("Histogram dims cannot be empty. Use dims=: to reduce over all dimensions."))

    all(d -> d in (1, 2, 3), dims_tuple) ||
        throw(ArgumentError("Histogram dims must be a subset of (1, 2, 3), but got dims=$dims."))

    return dims_tuple
end

function validate_histogram_method(method::Symbol)
    method ∈ (:sum, :integral, :average) ||
        throw(ArgumentError("Histogram currently only supports method = :sum, :integral, or :average."))
    return method
end

validate_histogram_method(method) =
    throw(ArgumentError("Histogram currently only supports method = :sum, :integral, or :average."))

function validate_histogram_architecture(arch)
    if isdefined(Oceananigans, :DistributedComputations) &&
       arch isa Oceananigans.DistributedComputations.Distributed
        throw(ArgumentError("Histogram does not support distributed architectures yet."))
    end

    return nothing
end

function validate_histogram_bins(bins::NamedTuple)
    length(bins) >= 1 ||
        throw(ArgumentError("Histogram requires bins to be a NamedTuple with at least one entry."))

    edge_values = map(collect, values(bins))
    edge_names = keys(bins)

    for (name, edges) in zip(edge_names, edge_values)
        length(edges) >= 2 ||
            throw(ArgumentError("Histogram bin edges for `$name` must contain at least two values."))

        for n in 2:length(edges)
            edges[n] > edges[n-1] ||
                throw(ArgumentError("Histogram bin edges for `$name` must be strictly increasing."))
        end
    end

    return NamedTuple{edge_names}(edge_values)
end

validate_histogram_bins(bins) =
    throw(ArgumentError("Histogram requires bins to be a NamedTuple, or a single edge vector for 1D histograms."))

function validate_histogram_bins_2d(bins)
    bins = validate_histogram_bins(bins)
    length(bins) == 2 ||
        throw(ArgumentError("2D Histogram requires bins to be a NamedTuple with exactly two entries."))

    return bins
end

function validate_histogram_bins_1d(bins::NamedTuple)
    bins = validate_histogram_bins(bins)
    length(bins) == 1 ||
        throw(ArgumentError("1D Histogram requires bins to be a NamedTuple with exactly one entry."))

    return bins
end

validate_histogram_bins_1d(edges) = validate_histogram_bins_1d((a = edges,))

function reorder_histogram_bins_for_named_operands(bins::NamedTuple, operand_names::Tuple{Symbol, Symbol})
    bins = validate_histogram_bins_2d(bins)
    first_name, second_name = operand_names

    first_name ∈ keys(bins) ||
        throw(ArgumentError("Histogram bins are missing key `$first_name` required by named operand mapping."))

    second_name ∈ keys(bins) ||
        throw(ArgumentError("Histogram bins are missing key `$second_name` required by named operand mapping."))

    return NamedTuple{operand_names}((getproperty(bins, first_name),
                                      getproperty(bins, second_name)))
end

function reorder_histogram_bins_for_named_operand(bins::NamedTuple, operand_name::Symbol)
    bins = validate_histogram_bins_1d(bins)

    operand_name ∈ keys(bins) ||
        throw(ArgumentError("Histogram bins are missing key `$operand_name` required by named operand mapping."))

    return NamedTuple{(operand_name,)}((getproperty(bins, operand_name),))
end

@inline histogram_integral_metric(a, dims) =
    grid_metric_operation(instantiated_location(a), reduction_grid_metric(dims), a.grid)

function validate_histogram_weights(weights::Symbol, method::Symbol, dims, a::AbstractField)
    if method === :sum
        if weights === :count
            return HistogramCountWeights()
        else
            throw(ArgumentError("Unsupported Histogram weights = $weights for method=:sum. Use :count or an AbstractField."))
        end
    elseif method === :integral
        metric = histogram_integral_metric(a, dims)

        if weights === :count
            return HistogramIntegralCountWeights(metric)
        else
            throw(ArgumentError("Unsupported Histogram weights = $weights for method=:integral. Use :count or an AbstractField."))
        end
    else # method === :average
        metric = histogram_integral_metric(a, dims)

        if weights === :count
            return HistogramAverageCountWeights(metric)
        else
            throw(ArgumentError("Unsupported Histogram weights = $weights for method=:average. Use :count or an AbstractField."))
        end
    end
end

function validate_histogram_weights(weights::AbstractField, method::Symbol, dims, a::AbstractField)
    weights.grid == a.grid ||
        throw(ArgumentError("Histogram weight field must be on the same grid as the operand."))

    instantiated_location(weights) == instantiated_location(a) ||
        throw(ArgumentError("Histogram weight field must have the same location as the operand."))

    if method === :sum
        return HistogramFieldWeights(weights)
    elseif method === :integral
        metric = histogram_integral_metric(a, dims)
        return HistogramIntegralFieldWeights(weights, metric)
    else
        metric = histogram_integral_metric(a, dims)
        return HistogramAverageFieldWeights(weights, metric)
    end
end

validate_histogram_weights(weights, method::Symbol, dims, a::AbstractField) =
    throw(ArgumentError("Unsupported Histogram weights = $weights for method=$method."))

function validate_histogram_weights(weights, method::Symbol, dims, a::AbstractField, b::AbstractField)
    validate_histogram_operands(a, b)
    return validate_histogram_weights(weights, method, dims, a)
end

@inline function histogram_eltype(a, b, bins, ::HistogramCountWeights)
    edges1, edges2 = values(bins)
    return promote_type(Float64, float(eltype(a)), float(eltype(b)), float(eltype(edges1)), float(eltype(edges2)))
end

@inline function histogram_eltype(a, b, bins, weights::HistogramFieldWeights)
    edges1, edges2 = values(bins)
    return promote_type(float(eltype(a)),
                        float(eltype(b)),
                        float(eltype(edges1)),
                        float(eltype(edges2)),
                        float(eltype(weights.field)))
end

@inline function histogram_eltype(a, b, bins, weights::HistogramIntegralCountWeights)
    edges1, edges2 = values(bins)
    return promote_type(float(eltype(a)),
                        float(eltype(b)),
                        float(eltype(edges1)),
                        float(eltype(edges2)),
                        float(eltype(weights.metric)))
end

@inline function histogram_eltype(a, b, bins, weights::HistogramIntegralFieldWeights)
    edges1, edges2 = values(bins)
    return promote_type(float(eltype(a)),
                        float(eltype(b)),
                        float(eltype(edges1)),
                        float(eltype(edges2)),
                        float(eltype(weights.field)),
                        float(eltype(weights.metric)))
end

@inline function histogram_eltype(a, b, bins, weights::HistogramAverageCountWeights)
    edges1, edges2 = values(bins)
    return promote_type(float(eltype(a)),
                        float(eltype(b)),
                        float(eltype(edges1)),
                        float(eltype(edges2)),
                        float(eltype(weights.metric)))
end

@inline function histogram_eltype(a, b, bins, weights::HistogramAverageFieldWeights)
    edges1, edges2 = values(bins)
    return promote_type(float(eltype(a)),
                        float(eltype(b)),
                        float(eltype(edges1)),
                        float(eltype(edges2)),
                        float(eltype(weights.field)),
                        float(eltype(weights.metric)))
end

@inline function histogram_eltype(a, bins, ::HistogramCountWeights)
    edges1 = first(values(bins))
    return promote_type(Float64, float(eltype(a)), float(eltype(edges1)))
end

@inline function histogram_eltype(a, bins, weights::HistogramFieldWeights)
    edges1 = first(values(bins))
    return promote_type(float(eltype(a)), float(eltype(edges1)), float(eltype(weights.field)))
end

@inline function histogram_eltype(a, bins, weights::HistogramIntegralCountWeights)
    edges1 = first(values(bins))
    return promote_type(float(eltype(a)), float(eltype(edges1)), float(eltype(weights.metric)))
end

@inline function histogram_eltype(a, bins, weights::HistogramIntegralFieldWeights)
    edges1 = first(values(bins))
    return promote_type(float(eltype(a)), float(eltype(edges1)), float(eltype(weights.field)), float(eltype(weights.metric)))
end

@inline function histogram_eltype(a, bins, weights::HistogramAverageCountWeights)
    edges1 = first(values(bins))
    return promote_type(float(eltype(a)), float(eltype(edges1)), float(eltype(weights.metric)))
end

@inline function histogram_eltype(a, bins, weights::HistogramAverageFieldWeights)
    edges1 = first(values(bins))
    return promote_type(float(eltype(a)), float(eltype(edges1)), float(eltype(weights.field)), float(eltype(weights.metric)))
end

function convert_histogram_bins_eltype(bins::NamedTuple, ::Type{FT}) where FT
    edges = map(edges -> convert(Vector{FT}, edges), values(bins))
    return NamedTuple{keys(bins)}(edges)
end
