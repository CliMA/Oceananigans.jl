module OutputWriters

export
    write_output!,
    FieldSlicer,
    JLD2OutputWriter,
    NetCDFOutputWriter,
    Checkpointer, restore_from_checkpoint,
    WindowedTimeAverage

using CUDA

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Models

using Oceananigans: AbstractOutputWriter
using Oceananigans.Fields: OffsetArray

Base.open(ow::AbstractOutputWriter) = nothing
Base.close(ow::AbstractOutputWriter) = nothing

include("output_writer_utils.jl")
include("field_slicer.jl")
include("fetch_output.jl")
include("windowed_time_average.jl")
include("jld2_output_writer.jl")
include("netcdf_output_writer.jl")
include("checkpointer.jl")

end
