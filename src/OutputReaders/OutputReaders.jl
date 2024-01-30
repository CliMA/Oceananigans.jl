module OutputReaders

export InMemory, OnDisk
export FieldTimeSeries, FieldDataset
export Cyclical, Linear, Clamp

abstract type AbstractDataBackend end

mutable struct InMemory{I} <: AbstractDataBackend 
    index_range :: I
end

"""
    InMemory(N=Colon())

Return a `backend` for `FieldTimeSeries` that stores
`N` fields from the field time series in memory.
The default is `N = :`, which stores all fields in memory.
"""
function InMemory(chunk_size::Int)
    index_range = UnitRange(1, chunk_size)
    return InMemory(index_range)
end

InMemory() = InMemory(Colon())

struct OnDisk <: AbstractDataBackend end


# Time extrapolation modes
struct Cyclical{T} # Cyclical in time
    Δt :: T # the cycle period will be tᴺ - t¹ + Δt where tᴺ is the last time and t¹ is the first time
end 

struct Linear end # linear extrapolation
struct Clamp end # clamp to nearest value


# validate_backend(::InMemory{Nothing}, data) = InMemory(collect(1:size(data, 4)))
# validate_backend(::OnDisk,   data)          = OnDisk()
# validate_backend(in_memory::InMemory, data) = in_memory

include("field_time_series.jl")
include("gpu_adapted_field_time_series.jl")
include("time_extrapolation.jl")
include("memory_allocated_field_time_series.jl")
include("on_disk_field_time_series.jl")
include("update_field_time_series.jl")
include("field_dataset.jl")

end # module

