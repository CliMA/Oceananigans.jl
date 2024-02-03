module OutputReaders

export InMemory, OnDisk
export FieldTimeSeries, FieldDataset
export Cyclical, Linear, Clamp

using Adapt

#####
##### Data backends for FieldTimeSeries
#####

abstract type AbstractDataBackend end

mutable struct InMemory{I} <: AbstractDataBackend 
    indices :: I
end

struct GPUAdaptedInMemory{I} <: AbstractDataBackend 
    indices :: I
end

Adapt.adapt_structure(to, backend::InMemory) = GPUAdaptedInMemory(backend.indices)

"""
    InMemory(N=:)

Return a `backend` for `FieldTimeSeries` that stores `N`
fields in memory. The default `N = :` stores all fields in memory.
"""
function InMemory(chunk_size::Int)
    chunk_size < 2 &&
        throw(ArgumentError("The chunk_size for InMemory backend cannot be less than 2."))

    index_range = 1:chunk_size
    indices = tuple(index_range...) # GPU-friendly tuple

    return InMemory(indices)
end

InMemory() = InMemory(Colon())

struct OnDisk <: AbstractDataBackend end

#####
##### Time extrapolation modes for FieldTimeSeries
#####

"""
    Cyclical(period=nothing)

Specifies cyclical FieldTimeSeries linear Time extrapolation.
If `period` is not specified, it is inferred from the `fts::FieldTimeSeries`
as

```julia
t = fts.times
Δt = t[end] - t[end-1]
period = t[end] - t[1] + Δt
```
"""
struct Cyclical{FT} # Cyclical in time
    period :: FT
end 

Cyclical() = Cyclical(nothing)


"""
    Linear()

Specifies FieldTimeSeries linear Time extrapolation.
"""
struct Linear end

"""
    Clamp()

Specifies FieldTimeSeries Time extrapolation that returns data from the nearest value.
"""
struct Clamp end # clamp to nearest value

include("field_time_series.jl")
include("gpu_adapted_field_time_series.jl")
include("field_time_series_indexing.jl")
include("memory_allocated_field_time_series.jl")
include("on_disk_field_time_series.jl")
include("update_field_time_series.jl")
include("field_dataset.jl")

end # module

