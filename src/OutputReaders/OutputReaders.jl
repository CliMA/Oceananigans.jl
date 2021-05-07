module OutputReaders

export FieldTimeSeries
export InMemory, OnDisk

abstract type AbstractDataBackend end

struct InMemory <: AbstractDataBackend end
struct OnDisk <: AbstractDataBackend end

include("output_reader_utils.jl")
include("field_time_series.jl")

end # module
