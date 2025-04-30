module OrthogonalSphericalShellGrids

# The only thing we need!
export TripolarGrid, RotatedLatitudeLongitudeGrid

import Oceananigans

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Architectures: device, on_architecture, AbstractArchitecture, CPU, GPU
using Oceananigans.BoundaryConditions
using Oceananigans.ImmersedBoundaries
using Oceananigans.Utils
using Oceananigans.BoundaryConditions: Zipper
using Oceananigans.Fields: index_binary_search, convert_to_0_360
using Oceananigans.Grids: RightConnected
using Oceananigans.Grids: R_Earth,
                          halo_size, spherical_area_quadrilateral,
                          lat_lon_to_cartesian, generate_coordinate, topology

using Oceananigans.Operators
using Oceananigans.Utils: get_cartesian_nodes_and_vertices

using Distances
using Adapt
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using OffsetArrays

const ZBC = BoundaryCondition{<:Zipper}

include("generate_tripolar_coordinates.jl")
include("tripolar_grid.jl")
include("tripolar_field_extensions.jl")
include("rotated_latitude_longitude_grid.jl")

# Distributed computations on a tripolar grid
include("distributed_tripolar_grid.jl")
include("distributed_zipper.jl")
include("distributed_zipper_north_tags.jl")

end # module
