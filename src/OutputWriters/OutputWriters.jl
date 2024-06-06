module OutputWriters

export
    JLD2OutputWriter, NetCDFOutputWriter, written_names,
    Checkpointer, WindowedTimeAverage, FileSizeLimit,
    TimeInterval, IterationInterval, WallTimeInterval, AveragedTimeInterval

using CUDA

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Models

using Oceananigans: AbstractOutputWriter
using Oceananigans.Grids: interior_indices
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval, instantiate
using Oceananigans.Utils: pretty_filesize

using OffsetArrays

import Oceananigans: write_output!, initialize!

Base.open(ow::AbstractOutputWriter) = nothing
Base.close(ow::AbstractOutputWriter) = nothing

include("output_writer_utils.jl")
include("fetch_output.jl")
include("windowed_time_average.jl")
include("output_construction.jl")
include("jld2_output_writer.jl")
include("netcdf_output_writer.jl")
include("checkpointer.jl")

function written_names(filename)
    field_names = String[]
    jldopen(filename, "r") do file 
        all_names = keys(file["timeseries"])
        field_names = filter(n -> n != "t", all_names)
    end
    return field_names
end

end # module

