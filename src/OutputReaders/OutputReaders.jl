module OutputReaders

export InMemory, OnDisk
export FieldTimeSeries, FieldDataset
export Cyclical, Linear, Clamp


#####
##### Data backends for FieldTimeSeries
#####

abstract type AbstractDataBackend end

mutable struct InMemory{I} <: AbstractDataBackend 
    index_range :: I
end

"""
    InMemory(N=:)

Return a `backend` for `FieldTimeSeries` that stores `N`
fields in memory. The default `N = :` stores all fields in memory.
"""
function InMemory(chunk_size::Int)
    index_range = UnitRange(1, chunk_size)
    return InMemory(index_range)
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
include("time_extrapolation.jl")
include("memory_allocated_field_time_series.jl")
include("on_disk_field_time_series.jl")
include("update_field_time_series.jl")
include("field_dataset.jl")

end # module

