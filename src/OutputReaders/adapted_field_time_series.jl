using Adapt

struct AdaptedFieldTimeSeries{T, D, χ} <: AbstractArray{T, 4}
                   data :: D
                  times :: χ

    function AdaptedFieldTimeSeries{T}(data::D,
                                      times::χ) where {D, χ}
        T = eltype(data)
        return new{T, D, χ}(data, backend, times)
    end
end

Adapt.adapt_structure(to, fts::InMemoryFieldTimeSeries) = 
    AdaptedFieldTimeSeries{eltype(fts.grid)}(adapt(to, fts.data),
                                             adapt(to, fts.times))

