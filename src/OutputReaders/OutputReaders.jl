module OutputReaders

export FieldTimeSeries

abstract type AbstractDataBackend end

struct InMemory <: AbstractDataBackend end
struct OnDisk <: AbstractDataBackend end

include("field_time_series.jl")

end # module
