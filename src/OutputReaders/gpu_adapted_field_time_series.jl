using Adapt

struct GPUAdaptedFieldTimeSeries{LX, LY, LZ, TE, T, D, χ} <: AbstractArray{T, 4}
     data :: D
    times :: χ
    
    function GPUAdaptedFieldTimeSeries{LX, LY, LZ, TE, T}(data::D,
                                                          times::χ) where {LX, LY, LZ, TE, T, D, χ}
        return new{LX, LY, LZ, TE, T, D, χ}(data, times)
    end
end

Adapt.adapt_structure(to, fts::FieldTimeSeries{LX, LY, LZ, TE}) where {LX, LY, LZ, TE} = 
    GPUAdaptedFieldTimeSeries{LX, LY, LZ, TE, eltype(fts.grid)}(adapt(to, fts.data),
                                                                adapt(to, fts.times))

@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries) = lastindex(fts.data)
@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries, dim) = lastindex(fts.data, dim)
