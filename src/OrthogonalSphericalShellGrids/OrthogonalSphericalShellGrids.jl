module OrthogonalSphericalShellGrids

using Oceananigans

using Oceananigans: Face, Center
using Oceananigans.Architectures: device, on_architecture
using Oceananigans.BoundaryConditions
using Oceananigans.Fields: index_binary_search
using Oceananigans.Grids: RightConnected
using Oceananigans.Grids: R_Earth, 
                          halo_size, spherical_area_quadrilateral,
                          lat_lon_to_cartesian, generate_coordinate, topology
using Oceananigans.Architectures: AbstractArchitecture, CPU, architecture
using Oceananigans.Grids: OrthogonalSphericalShellGrid, R_Earth, halo_size
using Oceananigans.Utils: launch!, get_cartesian_nodes_and_vertices

using Oceananigans.Operators

using Distances: haversine
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using OffsetArrays
using Adapt

export RotatedLatitudeLongitudeGrid
export TripolarGrid, ZipperBoundaryCondition

include("rotated_latitude_longitude_grid.jl")

@inline convert_to_0_360(x) = ((x % 360) + 360) % 360

"""
    struct Tripolar{N, F, S}

A structure to represent a tripolar grid on a spherical shell.
"""
struct Tripolar{N, F, S}
    north_poles_latitude :: N
    first_pole_longitude :: F
    southernmost_latitude :: S
end

const TripolarGrid{FT, TX, TY, TZ, CZ, CC, FC, CF, FF, Arch} = OrthogonalSphericalShellGrid{FT, TX, TY, TZ, CZ, <:Tripolar, CC, FC, CF, FF, Arch}
const DistributedTripolarGrid{FT, TX, TY, TZ, CZ, CC, FC, CF, FF} = TripolarGrid{FT, TX, TY, TZ, CZ, CC, FC, CF, FF, <:Distributed}

Adapt.adapt_structure(to, t::Tripolar) = 
    Tripolar(Adapt.adapt(to, t.north_poles_latitude),
             Adapt.adapt(to, t.first_pole_longitude),
             Adapt.adapt(to, t.southernmost_latitude))

const TRG  = Union{TripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}}
const DTRG = Union{DistributedTripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:DistributedTripolarGrid}}
             
include("tripolar_grid_utils.jl")
include("zipper_boundary_condition.jl")
include("generate_tripolar_coordinates.jl")
include("tripolar_grid.jl")
include("tripolar_grid_extensions.jl")
include("distributed_tripolar_grid.jl")
include("with_halo.jl")

end # module
