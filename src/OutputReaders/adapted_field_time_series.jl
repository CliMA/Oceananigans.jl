using Adapt

struct GPUAdaptedFieldTimeSeries{T, LX, LY, LZ, D, χ} <: AbstractArray{T, 4}
                   data :: D
                  times :: χ

    function GPUAdaptedFieldTimeSeries{T, LX, LY, LZ}(data::D,
                                                     times::χ) where {T, LX, LY, LZ, D, χ}
        return new{T, LX, LY, LZ, D, χ}(data, backend, times)
    end
end

Adapt.adapt_structure(to, fts::FieldTimeSeries{LX, LY, LZ}) where {LX, LY, LZ} = 
    GPUAdaptedFieldTimeSeries{eltype(fts.grid), LX, LY, LZ}(adapt(to, fts.data),
                                                            adapt(to, fts.times))

@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries) = lastindex(fts.data)
@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries, dim) = lastindex(fts.data, dim)

const XYFTS = FieldTimeSeries{<:Any, <:Any, <:Any, Nothing}
const XZFTS = FieldTimeSeries{<:Any, <:Any, Nothing, <:Any}
const YZFTS = FieldTimeSeries{<:Any, Nothing, <:Any, <:Any}

const XYGPUFTS = GPUAdaptedFieldTimeSeries{<:Any, <:Any, <:Any, Nothing}
const XZGPUFTS = GPUAdaptedFieldTimeSeries{<:Any, <:Any, Nothing, <:Any}
const YZGPUFTS = GPUAdaptedFieldTimeSeries{<:Any, Nothing, <:Any, <:Any}

Base.getindex(fts::XYGPUFTS, i::Int, j::Int, n) = fts.data[i, j, 1, n]
Base.getindex(fts::XZGPUFTS, i::Int, k::Int, n) = fts.data[i, 1, k, n]
Base.getindex(fts::YZGPUFTS, j::Int, k::Int, n) = fts.data[1, j, k, n]

Base.getindex(fts::XYFTS, i::Int, j::Int, n) = fts.data[i, j, 1, n]
Base.getindex(fts::XZFTS, i::Int, k::Int, n) = fts.data[i, 1, k, n]
Base.getindex(fts::YZFTS, j::Int, k::Int, n) = fts.data[1, j, k, n]

# Only `getindex` for GPUAdaptedFieldTimeSeries, no need to `setindex`
Base.getindex(fts::GPUAdaptedFieldTimeSeries, i::Int, j::Int, k::Int, n::Int)    = fts.data[i, j, k, n]
Base.getindex(fts::GPUAdaptedFieldTimeSeries, i::Int, j::Int, k::Int, t::Number) = fts.data[i, j, k, Time(t)]

# Extend Linear time interpolation for GPUAdaptedFieldTimeSeries
function Base.getindex(fts::GPUAdaptedFieldTimeSeries, i::Int, j::Int, k::Int, time_index::Time)
    Ntimes = length(fts.times)
    time = time_index.time
    n₁, n₂ = index_binary_search(fts.times, time, Ntimes)

    # fractional index
    @inbounds n = (n₂ - n₁) / (fts.times[n₂] - fts.times[n₁]) * (time - fts.times[n₁]) + n₁
    fts_interpolated = getindex(fts, i, j, k, n₂) * (n - n₁) + getindex(fts, i, j, k, n₁) * (n₂ - n)

    # Don't interpolate if n = 0.
    return ifelse(n₁ == n₂, getindex(fts, i, j, k, n₁), fts_interpolated)
end
