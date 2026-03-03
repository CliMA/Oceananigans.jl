using Oceananigans.Fields: AbstractField, instantiated_location, compute_at!, indices
using Oceananigans.Grids: RectilinearGrid, Center, Bounded, Flat, topology, interior_indices
using Oceananigans.Architectures: architecture, on_architecture, CPU
using Oceananigans.Utils: KernelParameters, launch!, tupleit
using Oceananigans.Grids: inactive_cell

using KernelAbstractions: @kernel, @index, @atomic

abstract type AbstractHistogramWeights end

struct HistogramCountWeights <: AbstractHistogramWeights end

struct HistogramVolumeWeights{V} <: AbstractHistogramWeights
    volume :: V
end

struct HistogramFieldWeights{W} <: AbstractHistogramWeights
    field :: W
end

struct HistogramOperation{G, T, A, B, BN, E1, E2, W, H, C, K} <: AbstractOperation{Center, Center, Center, G, T}
    grid :: G
    a :: A
    b :: B
    bins :: BN
    edges1 :: E1
    edges2 :: E2
    weights :: W
    local_histogram :: H
    global_cache :: C
    launch_parameters :: K
    dims :: NTuple{3, Int}
    method :: Symbol
end

function HistogramOperation(grid::G, a::A, b::B, bins::BN, edges1::E1, edges2::E2,
                            weights::W, local_histogram::H, global_cache::C,
                            launch_parameters::K, dims::NTuple{3, Int}, method::Symbol) where {G, A, B, BN, E1, E2, W, H, C, K}
    T = eltype(global_cache)
    return HistogramOperation{G, T, A, B, BN, E1, E2, W, H, C, K}(grid, a, b, bins, edges1, edges2,
                                                                   weights, local_histogram, global_cache,
                                                                   launch_parameters, dims, method)
end

"""
    Histogram(a::AbstractField, b::AbstractField; bins=NamedTuple(), weights=:count, dims=(1, 2, 3), method=:sum)
    Histogram(fields::NamedTuple; bins=NamedTuple(), weights=:count, dims=(1, 2, 3), method=:sum)

Construct a 2D histogram operation from field operands `a` and `b`.

The result is an `AbstractOperation` that can be used with `Field(...)` and output writers,
for example `Field(Histogram(...))`.

MVP constraints:
- `bins` must be a `NamedTuple` with exactly two strictly-increasing edge vectors.
- `weights` must be one of `:count`, `:cell_volume` (or `:volume`), or an `AbstractField`
  on the same grid and location as `a` and `b`.
- `dims` must be `:` or `(1, 2, 3)`.
- `method` must be `:sum`.

Bin mapping behavior:
- `Histogram(a, b; bins=(..., ...))` maps bins by value order.
- `Histogram((name1=a, name2=b); bins=(...))` maps bins by key and then reorders
  to operand order. This allows bin tuple order to differ from operand order.

Example
=======

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> T = CenterField(grid); S = CenterField(grid);

julia> set!(T, (x, y, z) -> x + z); set!(S, (x, y, z) -> y - z);

julia> T_edges = collect(range(0.0, stop=2.0, length=5));

julia> S_edges = collect(range(-1.0, stop=2.0, length=7));

julia> h = Field(Histogram(T, S; bins=(T=T_edges, S=S_edges), weights=:count));

julia> size(h)
(4, 6, 1)
```

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1));

julia> T = CenterField(grid); S = CenterField(grid);

julia> set!(T, (x, y, z) -> x + z); set!(S, (x, y, z) -> y - z);

julia> T_edges = collect(range(0.0, stop=2.0, length=5));

julia> S_edges = collect(range(-1.0, stop=2.0, length=7));

julia> h_named = Field(Histogram((S=S, T=T); bins=(T=T_edges, S=S_edges), weights=:count));

julia> size(h_named)
(6, 4, 1)
```
"""
function Histogram(a::AbstractField, b::AbstractField;
                   bins = NamedTuple(),
                   weights = :count,
                   dims = (1, 2, 3),
                   method = :sum)

    validate_histogram_operands(a, b)

    dims = validate_histogram_dims(dims)
    method = validate_histogram_method(method)
    bins = validate_histogram_bins(bins)
    weights = validate_histogram_weights(weights, a, b)

    FT = histogram_eltype(a, b, bins, weights)
    bins = convert_histogram_bins_eltype(bins, FT)
    edges1_cpu, edges2_cpu = values(bins)

    nbin1 = length(edges1_cpu) - 1
    nbin2 = length(edges2_cpu) - 1

    # Keep histogram data on host for deterministic indexing/output while computing
    # local contributions on the operand architecture.
    histogram_grid = RectilinearGrid(CPU(), FT;
                                     size = (nbin1, nbin2),
                                     topology = (Bounded, Bounded, Flat),
                                     halo = (1, 1),
                                     x = edges1_cpu,
                                     y = edges2_cpu)

    arch = architecture(a.grid)
    edges1 = on_architecture(arch, edges1_cpu)
    edges2 = on_architecture(arch, edges2_cpu)

    local_histogram = zeros(arch, FT, nbin1, nbin2)
    global_cache = zeros(FT, nbin1, nbin2, 1)
    launch_parameters = KernelParameters(histogram_launch_indices(a, b, weights)...)

    return HistogramOperation(histogram_grid, a, b, bins, edges1, edges2, weights,
                              local_histogram, global_cache, launch_parameters,
                              dims, method)
end

function Histogram(fields::NamedTuple;
                   bins = NamedTuple(),
                   weights = :count,
                   dims = (1, 2, 3),
                   method = :sum)

    length(fields) == 2 ||
        throw(ArgumentError("Histogram(fields=...) requires exactly two named field operands."))

    operand_names = keys(fields)
    a, b = values(fields)

    a isa AbstractField ||
        throw(ArgumentError("Histogram(fields=...) requires field operands, but `$(operand_names[1])` is $(typeof(a))."))

    b isa AbstractField ||
        throw(ArgumentError("Histogram(fields=...) requires field operands, but `$(operand_names[2])` is $(typeof(b))."))

    bins = reorder_histogram_bins_for_named_operands(bins, operand_names)
    return Histogram(a, b; bins, weights, dims, method)
end

Base.summary(::HistogramCountWeights) = ":count"
Base.summary(::HistogramVolumeWeights) = ":cell_volume"
Base.summary(w::HistogramFieldWeights) = "field ($(summary(w.field)))"

function Base.summary(op::HistogramOperation)
    nbin1, nbin2 = size(op.global_cache)[1:2]
    return "Histogram ($nbin1 × $nbin2 bins)"
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

@inline Base.getindex(op::HistogramOperation, i, j, k) = @inbounds op.global_cache[i, j, k]

function compute_at!(op::HistogramOperation, time)
    compute_at!(op.a, time)
    compute_at!(op.b, time)
    compute_histogram_weights_at!(op.weights, time)

    fill!(op.local_histogram, zero(eltype(op.local_histogram)))

    grid = op.a.grid
    arch = architecture(grid)

    launch!(arch, grid, op.launch_parameters, _accumulate_histogram!,
            op.local_histogram, op.a, op.b, op.edges1, op.edges2, grid, op.weights)

    local_histogram_cpu = on_architecture(CPU(), op.local_histogram)
    copyto!(view(op.global_cache, :, :, 1), local_histogram_cpu)
    histogram_all_reduce!(+, op.global_cache, arch)

    return nothing
end

@inline function histogram_all_reduce!(op, val, arch::Any)
    if isdefined(Oceananigans, :DistributedComputations)
        Oceananigans.DistributedComputations.all_reduce!(op, val, arch)
    end
    return val
end

@inline compute_histogram_weights_at!(::HistogramCountWeights, time) = nothing
@inline compute_histogram_weights_at!(weights::HistogramVolumeWeights, time) = compute_at!(weights.volume, time)
@inline compute_histogram_weights_at!(weights::HistogramFieldWeights, time) = compute_at!(weights.field, time)

@inline histogram_weight(i, j, k, grid, ::HistogramCountWeights) = 1
@inline histogram_weight(i, j, k, grid, weights::HistogramVolumeWeights) = @inbounds weights.volume[i, j, k]
@inline histogram_weight(i, j, k, grid, weights::HistogramFieldWeights) = @inbounds weights.field[i, j, k]

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

@kernel function _accumulate_histogram!(histogram, a, b, edges1, edges2, grid, weights)
    i, j, k = @index(Global, NTuple)

    if !inactive_cell(i, j, k, grid)
        @inbounds aᵢ = a[i, j, k]
        @inbounds bᵢ = b[i, j, k]

        ibin1 = find_histogram_bin(aᵢ, edges1)
        ibin2 = find_histogram_bin(bᵢ, edges2)

        in_range = (ibin1 > 0) & (ibin2 > 0)
        if in_range
            weight = convert(eltype(histogram), histogram_weight(i, j, k, grid, weights))
            @atomic histogram[ibin1, ibin2] += weight
        end
    end
end

@inline histogram_weight_operand(::HistogramCountWeights) = nothing
@inline histogram_weight_operand(weights::HistogramVolumeWeights) = weights.volume
@inline histogram_weight_operand(weights::HistogramFieldWeights) = weights.field

function histogram_launch_indices(a, b, weights)
    loc = instantiated_location(a)
    grid = a.grid

    idx1 = restricted_field_indices(a, 1, loc, grid)
    idx2 = restricted_field_indices(a, 2, loc, grid)
    idx3 = restricted_field_indices(a, 3, loc, grid)

    idx1 = intersect_histogram_ranges(idx1, restricted_field_indices(b, 1, loc, grid))
    idx2 = intersect_histogram_ranges(idx2, restricted_field_indices(b, 2, loc, grid))
    idx3 = intersect_histogram_ranges(idx3, restricted_field_indices(b, 3, loc, grid))

    weight_operand = histogram_weight_operand(weights)
    if !isnothing(weight_operand)
        idx1 = intersect_histogram_ranges(idx1, restricted_field_indices(weight_operand, 1, loc, grid))
        idx2 = intersect_histogram_ranges(idx2, restricted_field_indices(weight_operand, 2, loc, grid))
        idx3 = intersect_histogram_ranges(idx3, restricted_field_indices(weight_operand, 3, loc, grid))
    end

    return (idx1, idx2, idx3)
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
    dims_tuple = tupleit(dims)
    dims_tuple == (1, 2, 3) ||
        throw(ArgumentError("Histogram currently only supports dims = : or dims = (1, 2, 3)."))
    return dims_tuple
end

function validate_histogram_method(method::Symbol)
    method === :sum ||
        throw(ArgumentError("Histogram currently only supports method = :sum."))
    return method
end

validate_histogram_method(method) =
    throw(ArgumentError("Histogram currently only supports method = :sum."))

function validate_histogram_bins(bins::NamedTuple)
    length(bins) == 2 ||
        throw(ArgumentError("Histogram requires bins to be a NamedTuple with exactly two entries."))

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
    throw(ArgumentError("Histogram requires bins to be a NamedTuple with exactly two entries."))

function reorder_histogram_bins_for_named_operands(bins::NamedTuple, operand_names::Tuple{Symbol, Symbol})
    bins = validate_histogram_bins(bins)
    first_name, second_name = operand_names

    first_name ∈ keys(bins) ||
        throw(ArgumentError("Histogram bins are missing key `$first_name` required by named operand mapping."))

    second_name ∈ keys(bins) ||
        throw(ArgumentError("Histogram bins are missing key `$second_name` required by named operand mapping."))

    return NamedTuple{operand_names}((getproperty(bins, first_name),
                                      getproperty(bins, second_name)))
end

function validate_histogram_weights(weights::Symbol, a, b)
    if weights === :count
        return HistogramCountWeights()
    elseif weights === :cell_volume || weights === :volume
        volume_field = grid_metric_operation(instantiated_location(a), volume, a.grid)
        return HistogramVolumeWeights(volume_field)
    else
        throw(ArgumentError("Unsupported Histogram weights = $weights. Use :count, :cell_volume, :volume, or an AbstractField."))
    end
end

function validate_histogram_weights(weights::AbstractField, a, b)
    weights.grid == a.grid ||
        throw(ArgumentError("Histogram weight field must be on the same grid as the operands."))

    instantiated_location(weights) == instantiated_location(a) ||
        throw(ArgumentError("Histogram weight field must have the same location as the operands."))

    return HistogramFieldWeights(weights)
end

validate_histogram_weights(weights, a, b) =
    throw(ArgumentError("Unsupported Histogram weights = $weights. Use :count, :cell_volume, :volume, or an AbstractField."))

@inline function histogram_eltype(a, b, bins, ::HistogramCountWeights)
    edges1, edges2 = values(bins)
    return promote_type(float(eltype(a)), float(eltype(b)), float(eltype(edges1)), float(eltype(edges2)))
end

@inline function histogram_eltype(a, b, bins, ::HistogramVolumeWeights)
    edges1, edges2 = values(bins)
    return promote_type(float(eltype(a)),
                        float(eltype(b)),
                        float(eltype(edges1)),
                        float(eltype(edges2)),
                        float(eltype(a.grid)))
end

@inline function histogram_eltype(a, b, bins, weights::HistogramFieldWeights)
    edges1, edges2 = values(bins)
    return promote_type(float(eltype(a)),
                        float(eltype(b)),
                        float(eltype(edges1)),
                        float(eltype(edges2)),
                        float(eltype(weights.field)))
end

function convert_histogram_bins_eltype(bins::NamedTuple, ::Type{FT}) where FT
    edges = map(edges -> convert(Vector{FT}, edges), values(bins))
    return NamedTuple{keys(bins)}(edges)
end
