using Oceananigans.Fields: interpolator, _interpolate, fractional_indices
using Oceananigans.Architectures: architecture

import Oceananigans.Fields: interpolate

#####
##### Computation of time indices for interpolation
#####

# Simplest implementation, linear extrapolation if out-of-bounds
@inline interpolating_time_indices(::Linear, times, t) = time_index_binary_search(times, t)

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

    ñ, n₁, n₂ = time_index_binary_search(times, mod_t)

    cycling = ñ > 1 # we are _between_ tᴺ and t¹ + T
    cycled_indices   = (ñ - 1, Nt, 1)
    uncycled_indices = (ñ, n₁, n₂)

    return ifelse(cycling, cycled_indices, uncycled_indices)
end   

# Clamp mode if out-of-bounds, i.e get the neareast neighbor
@inline function interpolating_time_indices(::Clamp, times, t)
    n, n₁, n₂ = time_index_binary_search(times, t)

    beyond_indices    = (0, n₂, n₂) # Beyond the last time:  return n₂
    before_indices    = (0, n₁, n₁) # Before the first time: return n₁   
    unclamped_indices = (n, n₁, n₂) # Business as usual

    Nt = length(times)

    indices = ifelse(n + n₁ > Nt, beyond_indices,
              ifelse(n + n₁ < 1,  before_indices, unclamped_indices))

    return indices
end

@inline function time_index_binary_search(times, t)
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
    file = jldopen(fts.path)
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

@propagate_inbounds getindex(f::XYFTS, i::Int, j::Int, n::Time) = getindex(f, i, j, 1, n)
@propagate_inbounds getindex(f::XZFTS, i::Int, k::Int, n::Time) = getindex(f, i, 1, k, n)
@propagate_inbounds getindex(f::YZFTS, j::Int, k::Int, n::Time) = getindex(f, 1, j, k, n)

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
    ñ, n₁, n₂ = cpu_interpolating_time_indices(architecture(fts), fts.times, fts.time_indexing, time_index.time)

    if n₁ == n₂ # no interpolation needed
        return fts[n₁]
    end

    # Otherwise, make a Field representing a linear interpolation in time
    ψ₁ = fts[n₁]
    ψ₂ = fts[n₂]
    ψ̃ = Field(ψ₂ * ñ + ψ₁ * (1 - ñ))

    # Compute the field and return it
    return compute!(ψ̃)
end

#####
##### Linear time- and space-interpolation of a FTS
#####

@inline function interpolate(at_node, at_time_index::Time, from_fts::FlavorOfFTS, from_loc, from_grid)
    data = from_fts.data
    times = from_fts.times
    backend = from_fts.backend
    time_indexing = from_fts.time_indexing
    return interpolate(at_node, at_time_index, data, from_loc, from_grid, times, backend, time_indexing)
end

@inline function interpolate(at_node, at_time_index::Time, data::OffsetArray,
                             from_loc, from_grid, times, backend, time_indexing)

    at_time = at_time_index.time

    # Build space interpolators
    ii, jj, kk = fractional_indices(at_node, from_grid, from_loc...)

    ix = interpolator(ii)
    iy = interpolator(jj)
    iz = interpolator(kk)

    ñ, n₁, n₂ = interpolating_time_indices(time_indexing, times, at_time)

    Nt = length(times)
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

    target_times = map(Time, target_fts.times)

    launch!(arch, target_grid, size(target_fts),
            _interpolate_field_time_series!,
            target_fts.data, target_grid, target_location, target_times,
            source_fts, source_grid, source_location)

    fill_halo_regions!(target_fts)

    return nothing
end

@kernel function _interpolate_field_time_series!(target_fts, target_grid, target_location, target_times,
                                                 source_fts, source_grid, source_location)

    # 4D index, cool!
    i, j, k, n = @index(Global, NTuple)

    target_node = node(i, j, k, target_grid, target_location...)
    at_time     = @inbounds target_times[n]

    @inbounds target_fts[i, j, k, n] = interpolate(target_node, at_time,
                                                   source_fts, source_location, source_grid)
end

#####
##### FieldTimeSeries updating
#####

# Let's make sure `times` is available on the CPU. This is always the case 
# for ranges. if `times` is a vector that resides on the GPU, it has to be moved to the CPU for safe indexing. 
# TODO: Copying the whole array is a bit unclean, maybe find a way that avoids the penalty of allocating and copying memory.
# This would require refactoring `FieldTimeSeries` to include a cpu-allocated times array
cpu_interpolating_time_indices(::CPU, times, time_indexing, t, arch) = interpolating_time_indices(time_indexing, times, t)
cpu_interpolating_time_indices(::CPU, times::AbstractVector, time_indexing, t) = interpolating_time_indices(time_indexing, times, t)

function cpu_interpolating_time_indices(::GPU, times::AbstractVector, time_indexing, t) 
    cpu_times = on_architecture(CPU(), times)
    return interpolating_time_indices(time_indexing, cpu_times, t)
end

# Fallbacks that do nothing
update_field_time_series!(fts, time::Time) = nothing
update_field_time_series!(fts, n::Int) = nothing

# Update the `fts` to contain the time `time_index.time`.
# Linear extrapolation, simple version
function update_field_time_series!(fts::PartlyInMemoryFTS, time_index::Time)
    t = time_index.time
    ñ, n₁, n₂ = cpu_interpolating_time_indices(architecture(fts), fts.times, fts.time_indexing, t)
    return update_field_time_series!(fts, n₁, n₂)
end

function update_field_time_series!(fts::PartlyInMemoryFTS, n₁::Int, n₂=n₁)
    idxs = time_indices(fts)
    in_range = n₁ ∈ idxs && n₂ ∈ idxs

    if !in_range
        # Update backend
        Nm = length(fts.backend)
        start = n₁
        fts.backend = new_backend(fts.backend, start, Nm)
        set!(fts)
    end

    return nothing
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

