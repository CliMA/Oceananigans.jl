import Oceananigans.Fields: interpolate
using Oceananigans.Fields: interpolator, _interpolate, fractional_indices

const FlavorOfFTS{LX, LY, LZ, TE, K} =
    Union{GPUAdaptedFieldTimeSeries{LX, LY, LZ, TE, K},
                    FieldTimeSeries{LX, LY, LZ, TE, K}} where {LX, LY, LZ, TE, K} 

const CyclicalFTS{K} = FlavorOfFTS{<:Any, <:Any, <:Any, <:Cyclical, K} where K
const LinearFTS{K}   = FlavorOfFTS{<:Any, <:Any, <:Any, <:Linear, K} where K
const ClampFTS{K}    = FlavorOfFTS{<:Any, <:Any, <:Any, <:Clamp, K} where K

const TotallyInMemoryFTS = Union{FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:InMemory{Colon}},
                                 FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:GPUAdaptedInMemory{Colon}}}

const CyclicalChunkedFTS = Union{CyclicalFTS{<:InMemory{Tuple}}, CyclicalFTS{<:GPUAdaptedInMemory{Tuple}}}

# Reduced FTS
const XYFTS = FlavorOfFTS{<:Any, <:Any, Nothing, <:Any, <:Any}
const XZFTS = FlavorOfFTS{<:Any, Nothing, <:Any, <:Any, <:Any}
const YZFTS = FlavorOfFTS{Nothing, <:Any, <:Any, <:Any, <:Any}

#####
##### Reduced `getindex` with integers `(i, j, n)`
#####

@propagate_inbounds Base.getindex(fts::XYFTS, i::Int, j::Int, n) = fts[i, j, 1, n]
@propagate_inbounds Base.getindex(fts::XZFTS, i::Int, k::Int, n) = fts[i, 1, k, n]
@propagate_inbounds Base.getindex(fts::YZFTS, j::Int, k::Int, n) = fts[1, j, k, n]

#####
##### Underlying data index corresponding to time index `n :: int`
#####

@propagate_inbounds function Base.getindex(f::CyclicalChunkedFTS, i, j, k, n::Int)
    Ni = length(f.backend.indices)
    # Should find n₁ == n₂
    n₁, n₂ = index_binary_search(f.backend.indices, n, Ni)
    return f.data[i, j, k, n₁]
end

@propagate_inbounds Base.getindex(f::TotallyInMemoryFTS, i, j, k, n::Int) = f.data[i, j, k, n]

in_memory_time_index(time_extr,  index_range, n) = n - index_range[1] + 1
in_memory_time_index(time_extr,  ::Colon,     n) = n
in_memory_time_index(::Cyclical, ::Colon,     n) = n

function in_memory_time_index(::Cyclical, index_range, n) 
    Ni = length(index_range)
    # Should find n₁ == n₂
    n₁, n₂ = index_binary_search(index_range, n, Ni)
    return n₁
end

#####
##### Local `getindex` with integers `(i, j, k, n)`
#####

@propagate_inbounds Base.getindex(f::FlavorOfFTS, i, j, k, n::Int) =
    f.data[i, j, k, in_memory_time_index(f.time_extrapolation, f.backend.indices, n)]

#####
##### Local setindex! with integers `(i, j, k, n)` 
#####

@propagate_inbounds Base.setindex!(f::FlavorOfFTS, v, i, j, k, n::Int) =
    setindex!(f.data, v, i, j, k, n - f.backend.indices[1] + 1)

@propagate_inbounds function Base.setindex(f::CyclicalChunkedFTS, v, i, j, k, n::Int)
    Ni = length(f.backend.indices)
    # Should find n₁ == n₂
    n₁, n₂ = index_binary_search(f.backend.indices, n, Ni)
    return setindex!(f.data, v, i, j, k, n₁)
end    

@propagate_inbounds Base.setindex!(f::TotallyInMemoryFTS, v, i, j, k, n::Int) =
    setindex!(f.data, v, i, j, k, in_memory_time_index(f.time_extrapolation, f.backend.index_range, n))

#####
##### Local getindex with integers `(i, j, k)` and `n :: Time`
#####

# Valid for all flavors of FTS
@inline Base.getindex(fts::FlavorOfFTS, i::Int, j::Int, k::Int, time_index::Time) =
    interpolating_getindex(fts, i, j, k, time_index)

@inline function time_index_binary_search(fts, t)
    Nt = length(fts.times)

    # n₁ and n₂ are the index to interpolate inbetween and 
    # n is a fractional index where 0 ≤ n ≤ 1
    n₁, n₂ = index_binary_search(fts.times, t, Nt)

    @inbounds begin
        t₁ = fts.times[n₁]    
        t₂ = fts.times[n₂]    
    end

    n = (n₂ - n₁) / (t₂ - t₁) * (t - t₁)

    return n, n₁, n₂
end
    
# Simplest implementation, linear extrapolation if out-of-bounds
@inline interpolating_time_indices(fts::LinearFTS, t) = time_index_binary_search(fts, t)

# Cyclical implementation if out-of-bounds (wrap around the time-series)
@inline function interpolating_time_indices(fts::CyclicalFTS, t)
    times = fts.times
    Nt = length(times)
    t¹ = first(times) 
    tᴺ = last(times)

    T = fts.time_extrapolation.period
    Δt = T - (tᴺ - t¹)

    # Compute modulus of shifted time, then add shift back
    τ = t - t¹
    mod_τ = mod(τ, T)
    mod_t = mod_τ + t¹

    n, n₁, n₂ = time_index_binary_search(fts, mod_t)

    cycling = n > 1 # we are _between_ tᴺ and t¹ + T
    cycled_indices   = (1 - n, Nt, 1)
    uncycled_indices = (n, n₁, n₂)

    return ifelse(cycling, cycled_indices, uncycled_indices)
end   

# Clamp mode if out-of-bounds, i.e get the neareast neighbor
@inline function interpolating_time_indices(fts::ClampFTS, t)
    n, n₁, n₂ = time_index_binary_search(fts, t)

    beyond_indices    = (0, n₂, n₂) # Beyond the last time:  return n₂
    before_indices    = (0, n₁, n₁) # Before the first time: return n₁   
    unclamped_indices = (n, n₁, n₂) # Business as usual

    Nt = length(fts.times)

    indices = ifelse(n + n₁ > Nt, beyond_indices,
              ifelse(n + n₁ < 1,  before_indices, unclamped_indices))

    return indices
end

@inline function interpolating_getindex(fts, i, j, k, time_index)
    ñ, n₁, n₂ = interpolating_time_indices(fts, time_index.time)
    
    ψ₁ = getindex(fts, i, j, k, n₁)
    ψ₂ = getindex(fts, i, j, k, n₂)
    ψ̃ = ψ₂ * ñ + ψ₁ * (1 - ñ)

    # Don't interpolate if n₁ == n₂
    return ifelse(n₁ == n₂, ψ₁, ψ̃)
end

#####
##### Global `getindex` with `time_index :: Time`
#####

# Linear time interpolation
function Base.getindex(fts::FieldTimeSeries, time_index::Time)
    # Calculate fractional index (0 ≤ ñ ≤ 1)
    ñ, n₁, n₂ = interpolating_time_indices(fts, time_index.time)

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
    # Build space interpolators
    ii, jj, kk = fractional_indices(at_node, from_grid, from_loc...)

    ix = interpolator(ii)
    iy = interpolator(jj)
    iz = interpolator(kk)

    ñ, n₁, n₂ = interpolating_time_indices(from_fts, at_time_index.time)

    ψ₁ = _interpolate(from_fts, ix, iy, iz, n₁)
    ψ₂ = _interpolate(from_fts, ix, iy, iz, n₂)
    ψ̃ = ψ₂ * ñ + ψ₁ * (1 - ñ)

    # Don't interpolate if n₁ == n₂
    return ifelse(n₁ == n₂, ψ₁, ψ̃)
end

