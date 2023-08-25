module OutputReaders

export InMemory, OnDisk
export FieldTimeSeries, FieldDataset

abstract type AbstractDataBackend end

struct InMemory <: AbstractDataBackend end

# Not like this, maybe simplify between OnDisk and OnDiskData?
struct OnDisk <: AbstractDataBackend 
    path :: String
    name :: String
end

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
    data :: D
    index_range :: I
end

Base.summary(odd::OnDiskData)  = "OnDiskData($(odd.path), $(odd.name))"
Base.summary(odd::ChunkedData) = "ChunkedData($(odd.path), $(odd.name), with indices $(index_range))"

include("field_time_series.jl")
include("field_dataset.jl")

end # module
