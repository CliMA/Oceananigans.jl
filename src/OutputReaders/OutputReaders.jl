module OutputReaders

export InMemory, OnDisk
export FieldTimeSeries, FieldDataset

abstract type AbstractDataBackend end

struct InMemory <: AbstractDataBackend end
struct OnDisk <: AbstractDataBackend end

struct OnDiskData
    path :: String
    name :: String
end

Base.summary(odd::OnDiskData) = "OnDiskData($(odd.path), $(odd.name))"

include("field_time_series.jl")
include("field_dataset.jl")

end # module
