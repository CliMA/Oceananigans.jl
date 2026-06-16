"""
    OceananigansZarrExt

Extension that adds Zarr read/write support to Oceananigans.jl via [Zarr.jl](https://github.com/JuliaIO/Zarr.jl).

# Features

- `ZarrWriter`: saves model output to a Zarr store (`DirectoryStore`, `DictStore`, `S3Store`).
"""
module OceananigansZarrExt

using Zarr
using OrderedCollections: OrderedDict

using Oceananigans: AbstractModel
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: AbstractField, location, indices
import Oceananigans.Grids: grid
using Oceananigans.Grids:
    AbstractGrid, RectilinearGrid, LatitudeLongitudeGrid, OrthogonalSphericalShellGrid,
    Center, Face, Flat, Periodic, Bounded,
    RightCenterFolded, RightFaceFolded,
    StaticVerticalDiscretization, MutableVerticalDiscretization, AbstractVerticalCoordinate,
    grid, topology, halo_size, xspacings, yspacings, zspacings, λspacings, φspacings,
    λnodes, φnodes,
    parent_index_range, nodes, ξnodes, ηnodes, rnodes, validate_index, peripheral_node, inactive_node,
    topology, constructor_arguments, architecture,
    generate_coordinate, total_length, interior_indices
using Oceananigans.ImmersedBoundaries:
    ImmersedBoundaryGrid,
    GridFittedBoundary,
    GridFittedBottom,
    PartialCellBottom,
    GFBIBG, PCBIBG
using Oceananigans.OrthogonalSphericalShellGrids:
    TripolarGrid, RotatedLatitudeLongitudeGrid,
    ConformalCubedSpherePanelGrid, Tripolar, LatitudeLongitudeRotation,
    conformal_mapping_info
using Oceananigans.Architectures: CPU, GPU, architecture
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
    dimension_name_generator_free_surface,
    vertical_coordinate_name,
    add_schedule_metadata!, default_output_attributes
using Oceananigans.Utils:
    TimeInterval, IterationInterval, WallTimeInterval, materialize_schedule,
    prettykeys, pretty_filesize

import Oceananigans: initialize!, write_output!
import Oceananigans.OutputWriters: ZarrWriter

const c = Center()
const f = Face()

include("utils.jl")
include("dimensions.jl")
include("grid_reconstruction.jl")
include("zarr_writer.jl")
include("output_readers.jl")

end # module
