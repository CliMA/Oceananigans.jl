module OutputReaders

export InMemory, OnDisk
export FieldTimeSeries, FieldDataset

abstract type AbstractDataBackend end

struct InMemory <: AbstractDataBackend end
struct OnDisk <: AbstractDataBackend end

include("field_time_series.jl")
include("field_dataset.jl")

end # module
