"""
    OceananigansNCDatasetsExt

Extension that adds NetCDF (via NCDatasets.jl) read/write support to Oceananigans.jl.

# Features

- NetCDFWriter: Saves model output and metadata to NetCDF files.
- Grid reconstruction: Saves all grid construction and boundary info for accurate grid reconstruction.
- FieldTimeSeries: Loads time series from NetCDF files.
"""
module OceananigansNCDatasetsExt

import NCDatasets
using NCDatasets: AbstractDataset, NCDataset, defDim, defGroup, dimnames, name, sync

using Dates: AbstractTime, UTC, now, DateTime
using Printf: @sprintf
using OrderedCollections: OrderedDict
using Statistics: mean

import Oceananigans

using Oceananigans: prettytime, pretty_filesize, AbstractModel
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Architectures: Architectures, CPU, GPU, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Fields
using Oceananigans.Fields: Fields, AbstractField, data, interior, set!, Reduction, location, indices
using Oceananigans.Grids:
    Center, Face, grid, topology, nodes, constructor_arguments,
    architecture, generate_coordinate, interior_indices

# Aliased to avoid clashing with `Oceananigans.OutputReaders.new_data`, which is a
# different function (5-arg, for FieldTimeSeries data allocation).
import Oceananigans.Grids: new_data as allocate_grid_data
using Oceananigans.OrthogonalSphericalShellGrids:
    TripolarGrid, RotatedLatitudeLongitudeGrid,
    Tripolar, LatitudeLongitudeRotation,
    conformal_mapping_info
using Oceananigans.Grids: OrthogonalSphericalShellGrid

using OffsetArrays: OffsetArray
using Oceananigans.ImmersedBoundaries:
    ImmersedBoundaryGrid, GridFittedBottom, GridFittedBoundary, PartialCellBottom,
    CenterImmersedCondition, InterfaceImmersedCondition, bottom_height_field
using Oceananigans.Models: LagrangianParticles
using Oceananigans.OutputReaders:
    InMemoryFTS,
    time_indices,
    InMemory,
    Linear,
    time_indices_length,
    new_data,
    UnspecifiedBoundaryConditions,
    NetCDFPath
using Oceananigans.OutputWriters:
    auto_extension,
    output_averaging_schedule,
    show_averaging_schedule,
    WindowedTimeAverage,
    NoFileSplitting,
    update_file_splitting_schedule!,
    construct_output,
    time_average_outputs,
    fetch_output,
    convert_output,
    fetch_and_convert_output,
    show_array_type,
    default_dimension_attributes,
    gather_dimensions,
    gather_grid_metrics,
    gather_immersed_boundary,
    field_dimensions,
    field_auxiliary_coordinates,
    squeeze_reduced_dimensions,
    inflate_reduced_dimensions,
    materialize_serialized_output
using Oceananigans.Utils:
    materialize_schedule, versioninfo_with_gpu, oceananigans_versioninfo, prettykeys

import NCDatasets: defVar
import Oceananigans: initialize!, write_output!
import Oceananigans.OutputReaders: FieldTimeSeries, set_from_netcdf!
import Oceananigans.OutputWriters:
    NetCDFWriter,
    write_grid_reconstruction_data!,
    convert_for_netcdf,
    materialize_from_netcdf,
    reconstruct_grid,
    trilocation_dim_name,
    dimension_name_generator_free_surface

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
