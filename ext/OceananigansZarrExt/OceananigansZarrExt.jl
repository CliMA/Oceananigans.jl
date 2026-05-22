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
    AbstractGrid, RectilinearGrid, LatitudeLongitudeGrid,
    Center, Face, Flat, Periodic, Bounded,
    topology, constructor_arguments
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
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
    WindowedTimeAverage
using Oceananigans.Utils:
    TimeInterval, IterationInterval, WallTimeInterval, materialize_schedule,
    prettykeys, pretty_filesize

import Oceananigans: initialize!, write_output!
import Oceananigans.OutputWriters: ZarrWriter

include("zarr_writer.jl")
include("output_readers.jl")

end # module
