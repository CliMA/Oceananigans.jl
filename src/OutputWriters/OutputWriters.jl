module OutputWriters

export
    write_output, read_output,
    JLD2OutputWriter, FieldOutput, FieldOutputs,
    NetCDFOutputWriter, write_grid,
    Checkpointer, restore_from_checkpoint

using Oceananigans, Oceananigans.Grids

using Oceananigans: AbstractOutputWriter, @hascuda

@hascuda using CUDAnative, CuArrays

include("output_writer_utils.jl")
include("jld2_output_writer.jl")
include("netcdf_output_writer.jl")
include("checkpointer.jl")

end
