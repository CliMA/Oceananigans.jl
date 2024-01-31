struct GPUAdaptedFieldTimeSeries{LX, LY, LZ, TE, K, T, D, χ} <: AbstractArray{T, 4}
    data :: D
    times :: χ
    backend :: K
    time_extrapolation :: TE
    
    function GPUAdaptedFieldTimeSeries{LX, LY, LZ, T}(data::D,
                                                      times::χ,
                                                      backend::K,
                                                      time_extrapolation::TE) where {LX, LY, LZ, TE, K, T, D, χ}
        return new{LX, LY, LZ, TE, K, T, D, χ}(data, times, backend, time_extrapolation)
    end
end

Adapt.adapt_structure(to, fts::FieldTimeSeries{LX, LY, LZ}) where {LX, LY, LZ} = 
    GPUAdaptedFieldTimeSeries{LX, LY, LZ, eltype(fts.grid)}(adapt(to, fts.data),
                                                            adapt(to, fts.times),
                                                            adapt(to, fts.backend),
                                                            adapt(to, fts.time_extrapolation))

@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries) = lastindex(fts.data)
@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries, dim) = lastindex(fts.data, dim)
