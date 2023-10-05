module OutputReaders

export InMemory, OnDisk
export FieldTimeSeries, FieldDataset

abstract type AbstractDataBackend end

struct InMemory{I} <: AbstractDataBackend 
    path :: String
    name :: String
    index_range :: I
end

InMemory(; chunk_size = Colon()) = chunk_size isa Colon ? 
                                          InMemory("", "", chunk_size) :
                                          InMemory("", "", 1:chunk_size)

struct OnDisk <: AbstractDataBackend 
    path :: String
    name :: String
end

OnDisk() = OnDisk("", "")

regularize_backend(::InMemory, path, name, data) = InMemory(path, name, collect(1:size(data, 4)))
regularize_backend(::OnDisk,   path, name, data) = OnDisk(path, name)

include("field_time_series.jl")
include("memory_allocated_field_time_series.jl")
include("on_disk_field_time_series.jl")
include("adapted_field_time_series.jl")
include("update_field_time_series.jl")
include("field_dataset.jl")

end # module
