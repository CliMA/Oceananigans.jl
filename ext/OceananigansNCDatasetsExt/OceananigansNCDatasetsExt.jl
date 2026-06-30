"""
    OceananigansNCDatasetsExt

Extension that adds NetCDF (via NCDatasets.jl) read/write support to Oceananigans.jl.

# Features

- NetCDFWriter: Saves model output and metadata to NetCDF files.
- Grid reconstruction: Saves all grid construction and boundary info for accurate grid reconstruction.
- FieldTimeSeries: Loads time series from NetCDF files.
"""
module OceananigansNCDatasetsExt

using NCDatasets
using NCDatasets: AbstractDataset

using Dates: AbstractTime, UTC, now, DateTime
using Printf: @sprintf
using OrderedCollections: OrderedDict
using Statistics: mean

import Oceananigans

using Oceananigans: initialize!, prettytime, pretty_filesize, AbstractModel
using Oceananigans.AbstractOperations: KernelFunctionOperation, AbstractOperation
using Oceananigans.Architectures: CPU, GPU, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields
using Oceananigans.Fields: set!, Reduction, reduced_dimensions, reduced_location, location, indices
using Oceananigans.Grids:
    Center, Face, Flat, Periodic, Bounded,
    RightCenterFolded, RightFaceFolded,
    AbstractGrid, RectilinearGrid, LatitudeLongitudeGrid,
    StaticVerticalDiscretization, MutableVerticalDiscretization, AbstractVerticalCoordinate,
    grid, topology, halo_size, xspacings, yspacings, zspacings, λspacings, φspacings,
    λnodes, φnodes,
    parent_index_range, nodes, ξnodes, ηnodes, rnodes, validate_index, peripheral_node, inactive_node,
    constructor_arguments, architecture,
    generate_coordinate, total_length, interior_indices

# Aliased to avoid clashing with `Oceananigans.OutputReaders.new_data`, which is a
# different function (5-arg, for FieldTimeSeries data allocation).
import Oceananigans.Grids: new_data as allocate_grid_data
using Oceananigans.OrthogonalSphericalShellGrids:
    TripolarGrid, RotatedLatitudeLongitudeGrid,
    ConformalCubedSpherePanelGrid, Tripolar, LatitudeLongitudeRotation,
    conformal_mapping_info
using Oceananigans.Grids: OrthogonalSphericalShellGrid

using OffsetArrays: OffsetArray
using Oceananigans.ImmersedBoundaries:
    ImmersedBoundaryGrid, GridFittedBottom, GFBIBG, GridFittedBoundary, PartialCellBottom, PCBIBG,
    CenterImmersedCondition, InterfaceImmersedCondition, underlying_grid
using Oceananigans.Models: LagrangianParticles
using Oceananigans.OutputReaders:
    InMemoryFTS,
    time_indices,
    InMemory,
    OnDisk,
    Linear,
    time_indices_length,
    new_data,
    UnspecifiedBoundaryConditions,
    NetCDFPath
using Oceananigans.OutputWriters:
    auto_extension,
    output_averaging_schedule,
    show_averaging_schedule,
    AveragedTimeInterval,
    WindowedTimeAverage,
    NoFileSplitting,
    update_file_splitting_schedule!,
    construct_output,
    time_average_outputs,
    restrict_to_interior,
    fetch_output,
    convert_output,
    fetch_and_convert_output,
    show_array_type
using Oceananigans.Utils:
    TimeInterval, IterationInterval, WallTimeInterval, materialize_schedule,
    versioninfo_with_gpu, oceananigans_versioninfo, prettykeys, add_time_interval

import NCDatasets: defVar
import Oceananigans: write_output!
import Oceananigans.OutputReaders: FieldTimeSeries, set_from_netcdf!
import Oceananigans.OutputWriters:
    NetCDFWriter,
    write_grid_reconstruction_data!,
    convert_for_netcdf,
    materialize_from_netcdf,
    reconstruct_grid,
    trilocation_dim_name,
    add_grid_suffix,
    dimension_name_generator_free_surface,
    vertical_coordinate_name

const c = Center()
const f = Face()

#####
##### Include scripts
#####

include("utils.jl")
include("dimensions.jl")
include("grid_reconstruction.jl")
include("netcdf_writer.jl")
include("output_readers.jl")

end # module
