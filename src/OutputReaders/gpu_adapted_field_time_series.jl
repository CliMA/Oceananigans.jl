using Adapt

struct GPUAdaptedFieldTimeSeries{LX, LY, LZ, TE, T, D, χ} <: AbstractArray{T, 4}
    data :: D
    times :: χ
    time_extrapolation :: TE
    
    function GPUAdaptedFieldTimeSeries{LX, LY, LZ, T}(data::D,
                                                      times::χ,
                                                      time_extrapolation::TE) where {LX, LY, LZ, TE, T, D, χ}
        return new{LX, LY, LZ, TE, T, D, χ}(data, times, time_extrapolation)
    end
end

Adapt.adapt_structure(to, fts::FieldTimeSeries{LX, LY, LZ}) where {LX, LY, LZ} = 
    GPUAdaptedFieldTimeSeries{LX, LY, LZ, eltype(fts.grid)}(adapt(to, fts.data),
                                                            adapt(to, fts.times),
                                                            adapt(to, fts.time_extrapolation))

@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries) = lastindex(fts.data)
@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries, dim) = lastindex(fts.data, dim)
