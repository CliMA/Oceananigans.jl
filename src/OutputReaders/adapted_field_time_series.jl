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


# Making AdaptedFieldTimeSeries behave like Vector
Base.lastindex(fts::AdaptedFieldTimeSeries) = size(fts, 4)
Base.firstindex(fts::AdaptedFieldTimeSeries) = 1
Base.length(fts::AdaptedFieldTimeSeries) = size(fts, 4)

Base.getindex(fts::AdaptedFieldTimeSeries, i::Int, j::Int, k::Int, n::Int) = fts.data[i, j, k, n]

# Linear time interpolation
function Base.getindex(fts::AdaptedFieldTimeSeries, i::Int, j::Int, k::Int, time_index::Time)
    Ntimes = length(fts.times)
    time = time_index.time
    n₁, n₂ = index_binary_search(fts.times, time, Ntimes)

    # fractional index
    @inbounds n = (n₂ - n₁) / (fts.times[n₂] - fts.times[n₁]) * (time - fts.times[n₁]) + n₁
    fts_interpolated = getindex(fts, i, j, k, n₂) * (n - n₁) + getindex(fts, i, j, k, n₁) * (n₂ - n)

    # Don't interpolate if n = 0.
    return ifelse(n₁ == n₂, getindex(fts, i, j, k, n₁), fts_interpolated)
end
