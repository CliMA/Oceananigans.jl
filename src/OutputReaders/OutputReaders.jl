module OutputReaders

export FieldDataset
export FieldTimeSeries
export InMemory, OnDisk
export Cyclical, Linear, Clamp

using Oceananigans.Utils

include("field_time_series.jl")
include("field_time_series_indexing.jl")
include("set_field_time_series.jl")
include("field_time_series_reductions.jl")
include("show_field_time_series.jl")
include("extract_field_time_series.jl")

# Experimental
include("field_dataset.jl")

end # module

