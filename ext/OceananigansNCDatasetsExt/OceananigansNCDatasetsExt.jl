module OceananigansNCDatasetsExt

using NCDatasets

using Oceananigans.Fields

using Dates: AbstractTime, UTC, now, DateTime
using Printf: @sprintf
using OrderedCollections: OrderedDict
using SeawaterPolynomials: BoussinesqEquationOfState
using Statistics: mean

using Oceananigans: initialize!, prettytime, pretty_filesize, AbstractModel
using Oceananigans.Architectures: CPU, GPU, on_architecture
using Oceananigans.AbstractOperations: KernelFunctionOperation, AbstractOperation
using Oceananigans.BuoyancyFormulations: BuoyancyForce, BuoyancyTracer, SeawaterBuoyancy, LinearEquationOfState
using Oceananigans.Fields: Reduction, reduced_dimensions, reduced_location, location, indices

using Oceananigans.Grids:
    Center, Face, Flat, Periodic, Bounded,
    AbstractGrid, RectilinearGrid, LatitudeLongitudeGrid, StaticVerticalDiscretization,
    topology, halo_size, xspacings, yspacings, zspacings, λspacings, φspacings,
    parent_index_range, nodes, ξnodes, ηnodes, rnodes, validate_index, peripheral_node,
    constructor_arguments, architecture

using Oceananigans.ImmersedBoundaries:
    ImmersedBoundaryGrid, GridFittedBottom, GFBIBG, GridFittedBoundary, PartialCellBottom, PCBIBG,
    CenterImmersedCondition, InterfaceImmersedCondition

using Oceananigans.Models: ShallowWaterModel, LagrangianParticles
using Oceananigans.Utils:
    TimeInterval, IterationInterval, WallTimeInterval, materialize_schedule,
    versioninfo_with_gpu, oceananigans_versioninfo, prettykeys, add_time_interval
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

using Oceananigans.OutputReaders: InMemoryFTS, time_indices
using NCDatasets: AbstractDataset

import NCDatasets: defVar
import Oceananigans: write_output!
import Oceananigans.OutputWriters:
    NetCDFWriter,
    write_grid_reconstruction_data!,
    convert_for_netcdf,
    materialize_from_netcdf,
    reconstruct_grid,
    trilocation_dim_name,
    dimension_name_generator_free_surface

import Oceananigans.OutputReaders: FieldTimeSeries_from_netcdf, set_from_netcdf!

using Oceananigans.OutputReaders:
    FieldTimeSeries,
    InMemory,
    OnDisk,
    Linear,
    time_indices_length,
    new_data,
    UnspecifiedBoundaryConditions

using Oceananigans.Fields: set!

const c = Center()
const f = Face()
const BoussinesqSeawaterBuoyancy = SeawaterBuoyancy{FT, <:BoussinesqEquationOfState, T, S} where {FT, T, S}
const BuoyancyBoussinesqEOSModel = BuoyancyForce{<:BoussinesqSeawaterBuoyancy, g} where {g}

#####
##### Include scripts
#####

include("utils.jl")
include("dimensions.jl")
include("grid_reconstruction.jl")
include("netcdf_writer.jl")
include("output_readers.jl")

end # module
