module OrthogonalSphericalShellGrids

# The only things we need!
export TripolarGrid, ZipperBoundaryCondition

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Architectures: device, on_architecture
using Oceananigans.BoundaryConditions
using Oceananigans.Fields: index_binary_search
using Oceananigans.Grids: RightConnected
using Oceananigans.Grids: R_Earth, 
                          halo_size, spherical_area_quadrilateral,
                          lat_lon_to_cartesian, generate_coordinate, topology
using Oceananigans.Operators
using Oceananigans.Utils: get_cartesian_nodes_and_vertices

using Adapt 
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using OffsetArrays

@inline convert_to_0_360(x) = ((x % 360) + 360) % 360

include("generate_tripolar_coordinates.jl")
include("tripolar_grid.jl")
include("tripolar_field_extensions.jl")

end
