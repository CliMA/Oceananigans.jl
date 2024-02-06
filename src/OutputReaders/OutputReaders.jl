module OutputReaders

export InMemory, OnDisk
export FieldTimeSeries, FieldDataset
export Cyclical, Linear, Clamp

using Adapt

#####
##### Data backends for FieldTimeSeries
#####

abstract type AbstractDataBackend end

struct InMemory{S} <: AbstractDataBackend 
    start :: S
    size :: S
end

"""
    InMemory(size=nothing)

Return a `backend` for `FieldTimeSeries` that stores `size`
fields in memory. The default `size = nothing` stores all fields in memory.
"""
function InMemory(size::Int)
    size < 2 && throw(ArgumentError("The `size' for InMemory backend cannot be less than 2."))
    return InMemory(1, size)
end

InMemory() = InMemory(nothing, nothing)

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
include("field_time_series_indexing.jl")
include("memory_allocated_field_time_series.jl")
include("on_disk_field_time_series.jl")
include("update_field_time_series.jl")
include("show_field_time_series.jl")

# Experimental
include("field_dataset.jl")

end # module

