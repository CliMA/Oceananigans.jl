
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
    interpolating_get_index(fts, i, j, k, time_index)

@inline Base.getindex(fts::FieldTimeSeries, i::Int, j::Int, k::Int, time_index::Time) =
    interpolating_get_index(fts, i, j, k, time_index)

const CyclicFTS = Union{GPUAdaptedFieldTimeSeries{<:Any, <:Any, <:Any, <:Cyclic}, FieldTimeSeries{<:Any, <:Any, <:Any, <:Cyclic}}
const LinearFTS = Union{GPUAdaptedFieldTimeSeries{<:Any, <:Any, <:Any, <:Linear}, FieldTimeSeries{<:Any, <:Any, <:Any, <:Linear}}
const ClampFTS  = Union{GPUAdaptedFieldTimeSeries{<:Any, <:Any, <:Any, <:Clamp},  FieldTimeSeries{<:Any, <:Any, <:Any, <:Clamp}}
    
# Simplest implementation, linear extrapolation if out-of-bounds
@inline function interpolated_time_indices(n₁, n₂, ::LinearFTS, t, t₁, t₂, Nt)
    n = (n₂ - n₁) / (t₂ - t₁) * (t - t₁) + n₁
    n = n - n₁
    return n, n₁, n₂
end

# Cyclic implementation if out-of-bounds (wrap around the time-series)
# Note: Cyclic interpolation will not work if t - t₂ > t₂ - t₁
# or if t₁ - t > t₂ - t₁ (i.e. if we are skipping several Δt)
# to make that work we need to `update_field_time_series!`
@inline function interpolated_time_indices(n₁, n₂, ::CyclicFTS, t, t₁, t₂, Nt)
    n = (n₂ - n₁) / (t₂ - t₁) * (t - t₁) + n₁
    n, n₁, n₂ = ifelse(n > Nt, (n - n₂, n₂, 1),   # Beyond the last time:  circle around
                ifelse(n < 1,  (n₁ - n, n₁, Nt),  # Before the first time: circle around
                               (n - n₁, n₁, n₂))) # Business as usual
    return n, n₁, n₂
end

# Clamp mode if out-of-bounds, i.e get the neareast neighbor
@inline function interpolated_time_indices(n₁, n₂, ::ClampFTS, t, t₁, t₂, Nt)
    n = @inbounds (n₂ - n₁) / (t₂ - t₁) * (t - t₁) + n₁
    n, n₁, n₂ = ifelse(n > Nt, (0,      n₂, n₂),  # Beyond the last time:  return n₂
                ifelse(n < 1,  (0,      n₁, n₁),  # Before the first time: return n₁   
                               (n - n₁, n₁, n₂))) # Business as usual
    return n, n₁, n₂
end

# Linear time interpolation
function Base.getindex(fts::FieldTimeSeries, time_index::Time)
    Ntimes = length(fts.times)
    time = time_index.time
    n₁, n₂ = index_binary_search(fts.times, time, Ntimes)
    if n₁ == n₂ # no interpolation
        return fts[n₁]
    end

    t₁ = @inbounds fts.times[n₁]    
    t₂ = @inbounds fts.times[n₂]    
    
    # Calculate fractional index
    n, n₁, n₂ = interpolated_time_indices(n₁, n₂, fts, time, t₁, t₂, Ntimes)

    # Make a Field representing a linear interpolation in time
    time_interpolated_field = Field(fts[n₂] * n + fts[n₁] * (1 - n))

    # Compute the field and return it
    return compute!(time_interpolated_field)
end

# Linear time interpolation, used by FieldTimeSeries and GPUAdaptedFieldTimeSeries
@inline function interpolating_get_index(fts, i, j, k, time_index)
    Ntimes = length(fts.times)
    time = time_index.time
    n₁, n₂ = index_binary_search(fts.times, time, Ntimes)

    t₁ = @inbounds fts.times[n₁]    
    t₂ = @inbounds fts.times[n₂]    

    # n₁ and n₂ are the index to interpolate inbetween and 
    # n is a fractional index where 0 ≤ n ≤ 1
    n, n₁, n₂ = interpolated_time_indices(n₁, n₂, fts, time, t₁, t₂, Ntimes)

    interpolated_fts = getindex(fts, i, j, k, n₂) * n + getindex(fts, i, j, k, n₁) * (1 - n)

    # Don't interpolate if n₁ == n₂
    return ifelse(n₁ == n₂, getindex(fts, i, j, k, n₁), interpolated_fts)
end