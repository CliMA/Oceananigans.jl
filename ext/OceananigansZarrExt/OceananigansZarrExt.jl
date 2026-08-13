"""
    OceananigansZarrExt

Extension that adds Zarr read/write support to Oceananigans.jl via [Zarr.jl](https://github.com/JuliaIO/Zarr.jl).

# Features

- `ZarrWriter`: saves model output to a Zarr store (`DirectoryStore`, `DictStore`, `S3Store`).
"""
module OceananigansZarrExt

import Zarr
using OrderedCollections: OrderedDict

import Dates
using Dates: AbstractTime, UTC, now, DateTime
using Oceananigans: AbstractModel
using Oceananigans.Architectures: Architectures, CPU, GPU, architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: AbstractField, location, indices
import Oceananigans.Grids: grid
using Oceananigans.Grids:
    OrthogonalSphericalShellGrid, Center, Face, grid, topology,
    constructor_arguments, architecture, generate_coordinate, interior_indices

# Aliased to avoid clashing with `Oceananigans.OutputReaders.new_data`, which is a
# different function (5-arg, for FieldTimeSeries data allocation).
import Oceananigans.Grids: new_data as allocate_grid_data
using OffsetArrays: OffsetArray
using Oceananigans.ImmersedBoundaries:
    ImmersedBoundaryGrid,
    GridFittedBoundary,
    GridFittedBottom,
    PartialCellBottom
using Oceananigans.OrthogonalSphericalShellGrids:
    Tripolar, LatitudeLongitudeRotation,
    conformal_mapping_info
using Oceananigans.Models: LagrangianParticles
using Oceananigans.DistributedComputations:
    Distributed, DistributedGrid, global_barrier, mpi_rank, mpi_initialized,
    global_communicator, concatenate_local_sizes
import Oceananigans
using Oceananigans.OutputWriters:
    auto_extension,
    NoFileSplitting,
    update_file_splitting_schedule!,
    construct_output,
    time_average_outputs,
    output_averaging_schedule,
    show_averaging_schedule,
    show_array_type,
    trilocation_dim_name,
    fetch_and_convert_output,
    WindowedTimeAverage,
    add_grid_suffix,
    add_schedule_metadata!,
    default_output_attributes,
    default_dimension_attributes,
    gather_dimensions,
    gather_grid_metrics,
    gather_immersed_boundary,
    field_dimensions,
    field_auxiliary_coordinates,
    drop_reduced_dimensions,
    squeeze_reduced_dimensions,
    materialize_serialized_output
using Oceananigans.Utils:
    materialize_schedule,
    versioninfo_with_gpu, oceananigans_versioninfo,
    prettykeys

import Oceananigans: initialize!, write_output!
import Oceananigans.OutputWriters: ZarrWriter

const c = Center()
const f = Face()

include("utils.jl")
include("grid_reconstruction.jl")
include("zarr_writer.jl")
include("output_readers.jl")

end # module
