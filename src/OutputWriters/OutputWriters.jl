module OutputWriters

export
    JLD2Writer, NetCDFWriter, written_names,
    Checkpointer, checkpoint, WindowedTimeAverage, FileSizeLimit,
    TimeInterval, IterationInterval, WallTimeInterval, AveragedTimeInterval, AveragedSpecifiedTimes

using Oceananigans: AbstractOutputWriter
using Oceananigans.Architectures
using Oceananigans.Fields
using Oceananigans.Grids
using Oceananigans.Grids: interior_indices
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval, instantiate, pretty_filesize

using OffsetArrays

import Oceananigans: boundary_conditions, write_output!, initialize!

const c = Center()
const f = Face()

Base.open(::AbstractOutputWriter) = nothing
Base.close(::AbstractOutputWriter) = nothing

# Default fallback: most output writers don't need special initialization
initialize!(::AbstractOutputWriter, model) = nothing

include("output_writer_utils.jl")
include("fetch_output.jl")
include("averaged_specified_times.jl")
include("windowed_time_average.jl")
include("output_construction.jl")
include("jld2_writer.jl")
include("output_attributes.jl")
include("netcdf_writer.jl")
include("checkpointer.jl")

function written_names(filename)
    return jldopen(filename, "r") do file
        all_names = keys(file["timeseries"])
        filter(n -> n != "t", all_names)
    end
end

end # module
