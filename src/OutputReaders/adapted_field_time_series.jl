using Adapt

struct GPUAdaptedFieldTimeSeries{LX, LY, LZ, T, D, χ} <: AbstractArray{T, 4}
     data :: D
    times :: χ

    function GPUAdaptedFieldTimeSeries{LX, LY, LZ, T}(data::D,
                                                     times::χ) where {T, LX, LY, LZ, D, χ}
        return new{LX, LY, LZ, T, D, χ}(data, backend, times)
    end
end

Adapt.adapt_structure(to, fts::FieldTimeSeries{LX, LY, LZ}) where {LX, LY, LZ} = 
    GPUAdaptedFieldTimeSeries{LX, LY, LZ, eltype(fts.grid)}(adapt(to, fts.data),
                                                            adapt(to, fts.times))

@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries) = lastindex(fts.data)
@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries, dim) = lastindex(fts.data, dim)

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

# Only `getindex` for GPUAdaptedFieldTimeSeries, no need to `setindex`
Base.getindex(fts::GPUAdaptedFieldTimeSeries, i::Int, j::Int, k::Int, n::Int) = @inbounds fts.data[i, j, k, n]

# Extend Linear time interpolation for GPUAdaptedFieldTimeSeries
@inline Base.getindex(fts::GPUAdaptedFieldTimeSeries, i::Int, j::Int, k::Int, time_index::Time) =
    interpolating_get_index(fts, i, j, k, time_index)


