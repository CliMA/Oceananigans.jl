module OutputReaders

export FieldDataset
export FieldTimeSeries
export InMemory, OnDisk
export Cyclical, Linear, Clamp

"""
    auto_extension(filename, ext)

If `filename` ends in `ext`, return `filename`. Otherwise return `filename * ext`.
"""
function auto_extension(filename, ext)
    if endswith(filename, ext)
        return filename
    else
        return filename * ext
    end
end

include("field_time_series.jl")
include("field_time_series_indexing.jl")
include("set_field_time_series.jl")
include("field_time_series_reductions.jl")
include("show_field_time_series.jl")
include("extract_field_time_series.jl")

# Experimental
include("field_dataset.jl")

end # module

