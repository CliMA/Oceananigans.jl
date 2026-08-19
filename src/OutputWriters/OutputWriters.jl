module OutputWriters

export
    JLD2Writer, NetCDFWriter, ZarrWriter,
    Checkpointer, checkpoint,
    written_names,
    WindowedTimeAverage, AveragedSpecifiedTimes, FileSizeLimit,
    TimeInterval, IterationInterval, WallTimeInterval, AveragedTimeInterval

using DocStringExtensions: TYPEDSIGNATURES
using OffsetArrays: OffsetArrays, OffsetArray

using Oceananigans: Oceananigans, AbstractOutputWriter, boundary_conditions, write_output!
using Oceananigans.Architectures: Architectures, CPU, GPU
using Oceananigans.Fields: Fields, Field
using Oceananigans.Grids: Grids, AbstractGrid, Center, Face, LatitudeLongitudeGrid, RectilinearGrid,
                          interior_indices
using Oceananigans.Solvers: iteration
using Oceananigans.Utils: Utils, TimeInterval, IterationInterval, WallTimeInterval, instantiate,
                          pretty_filesize

const c = Center()
const f = Face()

Base.open(ow::AbstractOutputWriter) = nothing
Base.close(ow::AbstractOutputWriter) = nothing

# Default fallback: most output writers don't need special initialization
Oceananigans.initialize!(::AbstractOutputWriter, model) = nothing

include("output_writer_utils.jl")
include("fetch_output.jl")
include("averaged_specified_times.jl")
include("windowed_time_average.jl")
include("output_construction.jl")
include("jld2_writer.jl")
include("output_attributes.jl")
include("netcdf_writer.jl")
include("zarr_writer.jl")
include("checkpointer.jl")

function written_names(filename)
    return jldopen(filename, "r") do file
        all_names = keys(file["timeseries"])
        filter(n -> n != "t", all_names)
    end
end

end # module
