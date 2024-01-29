module OutputReaders

export InMemory, OnDisk
export FieldTimeSeries, FieldDataset
export Cyclic, Linear, Clamp

abstract type AbstractDataBackend end

mutable struct InMemory{I} <: AbstractDataBackend 
    index_range :: I
end

function InMemory(; chunk_size = Colon())
    index_range = if chunk_size isa Colon 
        Colon()
    else
        UnitRange(1, chunk_size)
    end

    return InMemory(index_range)
end

struct OnDisk <: AbstractDataBackend end

# validate_backend(::InMemory{Nothing}, data) = InMemory(collect(1:size(data, 4)))
# validate_backend(::OnDisk,   data)          = OnDisk()
# validate_backend(in_memory::InMemory, data) = in_memory

include("field_time_series.jl")
include("gpu_adapted_field_time_series.jl")
include("time_indexing.jl")
include("memory_allocated_field_time_series.jl")
include("on_disk_field_time_series.jl")
include("update_field_time_series.jl")
include("field_dataset.jl")

end # module

