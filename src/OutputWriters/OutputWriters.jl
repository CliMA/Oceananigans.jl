module OutputWriters

export
    JLD2OutputWriter, NetCDFOutputWriter,
    Checkpointer, restore_from_checkpoint,
    WindowedTimeAverage,
    TimeInterval, IterationInterval, WallTimeInterval, AveragedTimeInterval

using CUDA

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Models

using Oceananigans: AbstractOutputWriter
using Oceananigans.Grids: interior_indices
using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval

using OffsetArrays

import Oceananigans: write_output!

Base.open(ow::AbstractOutputWriter) = nothing
Base.close(ow::AbstractOutputWriter) = nothing

include("output_writer_utils.jl")
include("fetch_output.jl")
include("windowed_time_average.jl")
include("output_construction.jl")
include("jld2_output_writer.jl")
include("netcdf_output_writer.jl")
include("checkpointer.jl")

end
