import Oceananigans.Fields: interpolate
using Oceananigans.Fields: interpolator, _interpolate, fractional_indices

const XYFTS = FieldTimeSeries{<:Any, <:Any, Nothing}
const XZFTS = FieldTimeSeries{<:Any, Nothing, <:Any}
const YZFTS = FieldTimeSeries{Nothing, <:Any, <:Any}

const XYGPUFTS = GPUAdaptedFieldTimeSeries{<:Any, <:Any, Nothing}
const XZGPUFTS = GPUAdaptedFieldTimeSeries{<:Any, Nothing, <:Any}
const YZGPUFTS = GPUAdaptedFieldTimeSeries{Nothing, <:Any, <:Any}

# Handle `Nothing` locations to allow `getbc` to work
@propagate_inbounds Base.getindex(fts::XYGPUFTS, i::Int, j::Int, n) = fts[i, j, 1, n]
@propagate_inbounds Base.getindex(fts::XZGPUFTS, i::Int, k::Int, n) = fts[i, 1, k, n]
@propagate_inbounds Base.getindex(fts::YZGPUFTS, j::Int, k::Int, n) = fts[1, j, k, n]

@propagate_inbounds Base.getindex(fts::XYFTS, i::Int, j::Int, n) = fts[i, j, 1, n]
@propagate_inbounds Base.getindex(fts::XZFTS, i::Int, k::Int, n) = fts[i, 1, k, n]
@propagate_inbounds Base.getindex(fts::YZFTS, j::Int, k::Int, n) = fts[1, j, k, n]

@inline Base.getindex(fts::GPUAdaptedFieldTimeSeries, i::Int, j::Int, k::Int, time_index::Time) =
    interpolating_getindex(fts, i, j, k, time_index)

@inline Base.getindex(fts::FieldTimeSeries, i::Int, j::Int, k::Int, time_index::Time) =
    interpolating_getindex(fts, i, j, k, time_index)

const FlavorOfFTS{LX, LY, LZ, TE} =
    Union{GPUAdaptedFieldTimeSeries{LX, LY, LZ, TE},
                    FieldTimeSeries{LX, LY, LZ, TE}} where {LX, LY, LZ, TE} 

const CyclicalFTS{K} = FlavorOfFTS{<:Any, <:Any, <:Any, <:Cyclical, K} where K
const LinearFTS{K}   = FlavorOfFTS{<:Any, <:Any, <:Any, <:Linear, K} where K
const ClampFTS{K}    = FlavorOfFTS{<:Any, <:Any, <:Any, <:Clamp, K} where K

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

# Linear time interpolation, used by FieldTimeSeries and GPUAdaptedFieldTimeSeries
@inline function interpolating_getindex(fts, i, j, k, time_index)
    ñ, n₁, n₂ = interpolating_time_indices(fts, time_index.time)
    
    ψ₁ = getindex(fts, i, j, k, n₁)
    ψ₂ = getindex(fts, i, j, k, n₂)
    ψ̃ = ψ₂ * ñ + ψ₁ * (1 - ñ)

    # Don't interpolate if n₁ == n₂
    return ifelse(n₁ == n₂, ψ₁, ψ̃)
end

# Linear time- and space-interpolation
@inline function interpolate(at_node, at_time::Time, from_fts::FlavorOfFTS, from_loc, from_grid)
    # Build space interpolators
    ii, jj, kk = fractional_indices(at_node, from_grid, from_loc...)

    ix = interpolator(ii)
    iy = interpolator(jj)
    iz = interpolator(kk)

    ñ, n₁, n₂ = interpolating_time_indices(fts, at_time.time)

    ψ₁ = _interpolate(from_fts, ix, iy, iz, n₁)
    ψ₂ = _interpolate(from_fts, ix, iy, iz, n₂)
    ψ̃ = ψ₂ * ñ + ψ₁ * (1 - ñ)

    # Don't interpolate if n₁ == n₂
    return ifelse(n₁ == n₂, ψ₁, ψ̃)
end

