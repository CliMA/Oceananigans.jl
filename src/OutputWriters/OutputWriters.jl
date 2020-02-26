module OutputWriters

export
    write_output,
    JLD2OutputWriter, FieldOutput, FieldOutputs,
    NetCDFOutputWriter, write_grid_and_attributes,
    Checkpointer, restore_from_checkpoint

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Architectures

using Oceananigans: AbstractOutputWriter, @hascuda
using Oceananigans.Fields: OffsetArray

@hascuda using CUDAnative, CuArrays

include("output_writer_utils.jl")
include("jld2_output_writer.jl")
include("netcdf_output_writer.jl")
include("checkpointer.jl")

end
