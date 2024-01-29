module OutputReaders

export InMemory, OnDisk
export FieldTimeSeries, FieldDataset
export Cyclical, Linear, Clamp

abstract type AbstractDataBackend end

mutable struct InMemory{I} <: AbstractDataBackend 
    index_range :: I

    function InMemory(index_range=Colon())
        I = typeof(index_range)
        return new{I}(index_range)
    end
end

function InMemory(chunk_size::Int)
    index_range = UnitRange(1, chunk_size)
    return InMemory(index_range)
end

# `Ntimes` time steps in memory
InMemory(Ntimes::Int) = InMemory(UnitRange(1, Ntimes))

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

