module OutputWriters

export
    write_output,
    JLD2OutputWriter, FieldOutput, FieldOutputs,
    NetCDFOutputWriter, write_grid_and_attributes,
    Checkpointer, restore_from_checkpoint

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
include("jld2_output_writer.jl")
include("netcdf_output_writer.jl")
include("checkpointer.jl")

end
