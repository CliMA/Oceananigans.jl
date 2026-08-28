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
using Oceananigans.Fields: AbstractField, location, indices, interior
import Oceananigans.Grids: grid
using Oceananigans.Grids:
    OrthogonalSphericalShellGrid, Center, Face, grid,
    constructor_arguments, generate_coordinate
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
    deferred_output,
    deferred_outputs,
    placeholder_output,
    defer_schedule,
    WindowedTimeAverage,
    TimeDerivative,
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
    inflate_reduced_dimensions,
    construct_ossg_halo_padded_array,
    halo_fill_2d_metric,
    materialize_serialized_output
using Oceananigans.Utils:
    materialize_schedule, primary_actuation,
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
