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
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: Architectures, CPU, GPU, on_architecture
using Oceananigans.Fields: Fields, Field, AbstractField, location, reduced_dimensions, set!
using Oceananigans.Grids: Grids, AbstractGrid, Center, Face, Flat, LatitudeLongitudeGrid,
                          RectilinearGrid, StaticVerticalDiscretization, AbstractVerticalCoordinate,
                          ColumnEnsembleSize, Periodic, Bounded, FullyConnected,
                          LeftConnected, RightConnected, RightFaceFolded, RightCenterFolded,
                          LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded,
                          LeftConnectedRightCenterConnected, LeftConnectedRightFaceConnected,
                          grid, topology, halo_size, xspacings, yspacings, λspacings, φspacings,
                          λnodes, φnodes, validate_index, peripheral_node, inactive_node,
                          interior_indices
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary,
                                        GridFittedBottom, PartialCellBottom,
                                        CenterImmersedCondition, InterfaceImmersedCondition,
                                        GFBIBG, PCBIBG, bottom_height_field
using Oceananigans.OrthogonalSphericalShellGrids: OrthogonalSphericalShellGrid,
                                                   TripolarGrid, RotatedLatitudeLongitudeGrid,
                                                   ConformalCubedSpherePanelGrid,
                                                   LambertConformalConicGrid,
                                                   Tripolar, LatitudeLongitudeRotation,
                                                   CubedSphereConformalMapping,
                                                   LambertConformalConic
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
include("dimension_names.jl")
include("output_serialization.jl")
include("output_dimensions.jl")
include("output_grid_data.jl")
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
