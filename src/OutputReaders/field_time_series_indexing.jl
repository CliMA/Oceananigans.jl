using Oceananigans.Grids: _node
using Oceananigans.Fields: interpolator, _interpolate, FractionalIndices, flatten_node
using Oceananigans.Architectures: architecture
using Oceananigans.DistributedComputations: child_architecture, Distributed
using Adapt

import Oceananigans.Fields: interpolate

struct TimeInterpolator{T, N1, N2, N3}
    fractional_index :: T
    first_index  :: N1
    second_index :: N2
    length       :: N3
end

@inline TimeInterpolator(fts::FieldTimeSeries, t) =
    TimeInterpolator(fts.time_indexing, fts.times, t)

@inline function TimeInterpolator(time_indexing, times, t)
    ñ, n₁, n₂ = interpolating_time_indices(time_indexing, times, t)
    return TimeInterpolator(ñ, n₁, n₂, length(times))
end

Adapt.adapt_structure(to, ti::TimeInterpolator) =
    TimeInterpolator(Adapt.adapt(to, ti.fractional_index),
                     Adapt.adapt(to, ti.first_index),
                     Adapt.adapt(to, ti.second_index),
                     Adapt.adapt(to, ti.length))

#####
##### Computation of time indices for interpolation
#####

# Simplest implementation, linear extrapolation if out-of-bounds
@inline interpolating_time_indices(::Linear, times, t) = find_time_index(times, t)

# Cyclical implementation if out-of-bounds (wrap around the time-series)
@inline function interpolating_time_indices(ti::Cyclical, times, t)
    Nt = length(times)
    t¹ = first(times)
    tᴺ = last(times)

    T = ti.period
    Δt = T - (tᴺ - t¹)

    # Compute modulus of shifted time, then add shift back
    τ = t - t¹
    mod_τ = mod(τ, T)
    mod_t = mod_τ + t¹

    ñ, n₁, n₂ = find_time_index(times, mod_t)

    cycling = ñ > 1 # we are _between_ tᴺ and t¹ + T
    cycled_indices   = (ñ - 1, Nt, 1)
    uncycled_indices = (ñ, n₁, n₂)

    return ifelse(cycling, cycled_indices, uncycled_indices)
end

# Clamp mode if out-of-bounds, i.e get the neareast neighbor
@inline function interpolating_time_indices(::Clamp, times, t)
    ñ, n₁, n₂ = find_time_index(times, t)

    beyond_indices    = (0, n₂, n₂) # Beyond the last time:  return n₂
    before_indices    = (0, n₁, n₁) # Before the first time: return n₁
    unclamped_indices = (ñ, n₁, n₂) # Business as usual

    Nt = length(times)

    return ifelse(ñ + n₁ > Nt, beyond_indices,
           ifelse(ñ + n₁ < 1,  before_indices, unclamped_indices))
end

#####
##### fine_time_index
#####

@inline function find_time_index(times::StepRangeLen, t)
    n₂ = searchsortedfirst(times, t)

    Nt = length(times)
    n₂ = min(Nt, n₂) # cap
    n₁ = max(1, n₂ - 1)

    @inbounds begin
        t₁ = times[n₁]
        t₂ = times[n₂]
    end

    ñ = (t - t₁) / (t₂ - t₁)
    ñ = ifelse(n₂ == n₁, zero(ñ), ñ)

    return ñ, n₁, n₂
end

@inline function find_time_index(times, t)
    Nt = length(times)

    # n₁ and n₂ are the index to interpolate inbetween and
    # n is a fractional index where 0 ≤ n ≤ 1
    n₁, n₂ = index_binary_search(times, t, Nt)

    @inbounds begin
        t₁ = times[n₁]
        t₂ = times[n₂]
    end

    # "Fractional index" ñ ∈ (0, 1)
    ñ = (n₂ - n₁) / (t₂ - t₁) * (t - t₁)
    ñ = ifelse(n₂ == n₁, zero(ñ), ñ)

    return ñ, n₁, n₂
end

#####
##### `getindex` and `setindex` with integers `(i, j, n)`
#####

import Base: getindex

function getindex(fts::OnDiskFTS, n::Int)
    # Load data
    arch = architecture(fts)
    file = jldopen(fts.path; fts.reader_kw...)
    iter = keys(file["timeseries/t"])[n]
    raw_data = on_architecture(arch, file["timeseries/$(fts.name)/$iter"])
    close(file)

    # Wrap Field
    loc = location(fts)
    field_data = offset_data(raw_data, fts.grid, loc, fts.indices)

    return Field(loc, fts.grid;
                 indices = fts.indices,
                 boundary_conditions = fts.boundary_conditions,
                 data = field_data)
end

@propagate_inbounds getindex(f::FlavorOfFTS, i, j, k, n::Int) = getindex(f.data, i, j, k, memory_index(f, n))
@propagate_inbounds setindex!(f::FlavorOfFTS, v, i, j, k, n::Int) = setindex!(f.data, v, i, j, k, memory_index(f, n))

# Reduced FTS
const XYFTS = FlavorOfFTS{<:Any, <:Any, Nothing, <:Any, <:Any}
const XZFTS = FlavorOfFTS{<:Any, Nothing, <:Any, <:Any, <:Any}
const YZFTS = FlavorOfFTS{Nothing, <:Any, <:Any, <:Any, <:Any}

@propagate_inbounds getindex(f::XYFTS, i::Int, j::Int, n::Int) = getindex(f.data, i, j, 1, memory_index(f, n))
@propagate_inbounds getindex(f::XZFTS, i::Int, k::Int, n::Int) = getindex(f.data, i, 1, k, memory_index(f, n))
@propagate_inbounds getindex(f::YZFTS, j::Int, k::Int, n::Int) = getindex(f.data, 1, j, k, memory_index(f, n))

#####
##### Time interpolation / extrapolation
##### Local getindex with integers `(i, j, k)` and `n :: Time`
#####

# Valid for all flavors of FTS
@inline getindex(fts::FlavorOfFTS, i::Int, j::Int, k::Int, time_index::Time) =
    interpolating_getindex(fts, i, j, k, time_index)

@inline function interpolating_getindex(fts, i, j, k, time_index)
    ñ, n₁, n₂ = interpolating_time_indices(fts.time_indexing, fts.times, time_index.time)

    @inbounds begin
        ψ₁ = getindex(fts, i, j, k, n₁)
        ψ₂ = getindex(fts, i, j, k, n₂)
    end

    ψ̃ = ψ₂ * ñ + ψ₁ * (1 - ñ)

    # Don't interpolate if n₁ == n₂.
    return ifelse(n₁ == n₂, ψ₁, ψ̃)
end

#####
##### Global `getindex` with `time_index :: Time`
#####

# Linear time interpolation
function Base.getindex(fts::FieldTimeSeries, time_index::Time)
    # Calculate fractional index (0 ≤ ñ ≤ 1)
    indices = cpu_interpolating_time_indices(architecture(fts), fts.times, fts.time_indexing, time_index.time)
    ñ = indices.fractional_index
    n₁ = indices.first_index
    n₂ = indices.second_index

    if n₁ == n₂ # no interpolation needed
        return fts[n₁]
    end

    # Otherwise, make a Field representing a linear interpolation in time
    # Make sure both n₁ and n₂ are in memory by first retrieving n₂ and then n₁
    update_field_time_series!(fts, n₁, n₂)

    ψ₂ = fts[n₂]
    ψ₁ = fts[n₁]
    ψ̃  = Field(ψ₂ * ñ + ψ₁ * (1 - ñ))

    # Compute the field and return it
    return compute!(ψ̃)
end

#####
##### Linear time- and space-interpolation of a FTS
#####

@inline function interpolate(to_node, to_time_index::Time, from_fts::FlavorOfFTS, from_loc, from_grid)
    data = from_fts.data
    times = from_fts.times
    backend = from_fts.backend
    time_indexing = from_fts.time_indexing
    return interpolate(to_node, to_time_index, data, from_loc, from_grid, times, backend, time_indexing)
end

@inline function interpolate(to_node, to_time_index::Time, data::OffsetArray,
                             from_loc, from_grid, times, backend, time_indexing)

    to_time = to_time_index.time
    interp = TimeInterpolator(time_indexing, times, to_time)
    return interpolate(to_node, interp, data, from_loc, from_grid, backend, time_indexing)
end

@inline function interpolate(to_node, time_indices::TimeInterpolator, data::OffsetArray,
                             from_loc, from_grid, backend, time_indexing)

    # Build space interpolators
    to_node = flatten_node(to_node...)

    if topology(from_grid) === (Flat, Flat, Flat)
        fi = FractionalIndices(nothing, nothing, nothing)
    else
        fi = FractionalIndices(to_node, from_grid, from_loc...)
    end

    return interpolate(fi, time_indices, data, backend, time_indexing)
end

@inline function interpolate(fi::FractionalIndices, time_indices::TimeInterpolator,
                             data::OffsetArray, backend, time_indexing)

    ñ  = time_indices.fractional_index
    n₁ = convert(Int, time_indices.first_index)
    n₂ = convert(Int, time_indices.second_index)
    Nt = convert(Int, time_indices.length)

    ix = interpolator(fi.i)
    iy = interpolator(fi.j)
    iz = interpolator(fi.k)

    m₁ = memory_index(backend, time_indexing, Nt, n₁)
    m₂ = memory_index(backend, time_indexing, Nt, n₂)

    ψ₁ = _interpolate(data, ix, iy, iz, m₁)
    ψ₂ = _interpolate(data, ix, iy, iz, m₂)
    ψ̃ = ψ₂ * ñ + ψ₁ * (1 - ñ)

    # Don't interpolate if n₁ == n₂
    return ifelse(n₁ == n₂, ψ₁, ψ̃)
end

function interpolate!(target_fts::FieldTimeSeries, source_fts::FieldTimeSeries)

    target_grid = target_fts.grid
    source_grid = source_fts.grid

    @assert architecture(target_grid) == architecture(source_grid)
    arch = architecture(target_grid)

    # Make locations
    source_location = map(instantiate, location(source_fts))
    target_location = map(instantiate, location(target_fts))

    launch!(arch, target_grid, size(target_fts),
            _interpolate_field_time_series!,
            target_fts.data, target_grid, target_location, target_fts.times,
            source_fts, source_grid, source_location)

    fill_halo_regions!(target_fts)

    return nothing
end

@kernel function _interpolate_field_time_series!(target_fts, target_grid, target_location, target_times,
                                                 source_fts, source_grid, source_location)

    # 4D index, cool!
    i, j, k, n = @index(Global, NTuple)

    target_node = _node(i, j, k, target_grid, target_location...)
    to_time     = @inbounds Time(target_times[n])

    @inbounds target_fts[i, j, k, n] = interpolate(target_node, to_time,
                                                   source_fts, source_location, source_grid)
end

#####
##### FieldTimeSeries updating
#####

# Let's make sure `times` is available on the CPU. This is always the case
# for ranges. if `times` is a vector that resides on the GPU, it has to be moved to the CPU for safe indexing.
# TODO: Copying the whole array is a bit unclean, maybe find a way that avoids the penalty of allocating and copying memory.
# This would require refactoring `FieldTimeSeries` to include a cpu-allocated times array
cpu_interpolating_time_indices(::CPU, times, time_indexing, t) = TimeInterpolator(time_indexing, times, t)
cpu_interpolating_time_indices(::CPU, times::AbstractVector, time_indexing, t) = TimeInterpolator(time_indexing, times, t)

function cpu_interpolating_time_indices(::GPU, times::AbstractVector, time_indexing, t)
    cpu_times = on_architecture(CPU(), times)
    return TimeInterpolator(time_indexing, cpu_times, t)
end

cpu_interpolating_time_indices(arch::Distributed, args...) = cpu_interpolating_time_indices(child_architecture(arch), args...)

# Fallbacks that do nothing
update_field_time_series!(fts, time::Time) = nothing
update_field_time_series!(fts, n₁::Int, n₂=n₁) = nothing

# Update the `fts` to contain the time `time_index.time`.
# Linear extrapolation, simple version
function update_field_time_series!(fts::PartlyInMemoryFTS, time_index::Time)
    t = time_index.time
    indices = cpu_interpolating_time_indices(architecture(fts), fts.times, fts.time_indexing, t)
    n₁ = indices.first_index
    n₂ = indices.second_index
    return update_field_time_series!(fts, n₁, n₂)
end

function update_field_time_series!(fts::PartlyInMemoryFTS, n₁::Int, n₂=n₁)
    in_range = in_time_range(fts, fts.time_indexing, n₁, n₂)

    if !in_range
        # Update backend
        Nm = length(fts.backend)
        start = n₁
        fts.backend = new_backend(fts.backend, start, Nm)
        set!(fts)
        fill_halo_regions!(fts)
    end

    return nothing
end

function in_time_range(fts, time_indexing, n₁, n₂)
    idxs = time_indices(fts)
    return n₁ ∈ idxs && n₂ ∈ idxs
end

function in_time_range(fts, ::Union{Clamp, Linear}, n₁, n₂)
    Nt = length(fts.times)
    idxs = time_indices(fts)
    in_range_1 = n₁ ∈ idxs || n₁ > Nt
    in_range_2 = n₂ ∈ idxs || n₂ > Nt
    return in_range_1 && in_range_2
end

# If `n` is not in memory, getindex automatically updates the data in memory
# so that `n` is the first index available.
function getindex(fts::InMemoryFTS, n::Int)
    update_field_time_series!(fts, n)

    m = memory_index(fts, n)
    underlying_data = view(parent(fts), :, :, :, m)
    data = offset_data(underlying_data, fts.grid, location(fts), fts.indices)

    return Field(location(fts), fts.grid; data, fts.boundary_conditions, fts.indices)
end
