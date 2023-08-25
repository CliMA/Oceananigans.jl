module OutputReaders

export InMemory, OnDisk
export FieldTimeSeries, FieldDataset

abstract type AbstractDataBackend end

struct InMemory <: AbstractDataBackend end
struct OnDisk <: AbstractDataBackend end
struct Chunked <: AbstractDataBackend 
    chunk_size :: Int
end

Chunked(; chunk_size = 2) = Chunked(chunk_size)

struct OnDiskData
    path :: String
    name :: String
end

struct ChunkedData{D, I} 
    path :: String
    name :: String
    data_in_memory :: D
    index_range :: I
end

Base.summary(odd::OnDiskData)  = "OnDiskData($(odd.path), $(odd.name))"
Base.summary(odd::ChunkedData) = "ChunkedData($(odd.path), $(odd.name), with indices $(odd.index_range))"

include("field_time_series.jl")
include("field_dataset.jl")

end # module
