module OrthogonalSphericalShellGrids

# The only thing we need!
export TripolarGrid, RotatedLatitudeLongitudeGrid, ConformalCubedSpherePanelGrid

import Oceananigans
import Oceananigans.Architectures: on_architecture

using Oceananigans.Architectures: device, on_architecture, AbstractArchitecture, CPU, GPU
using Oceananigans.BoundaryConditions: BoundaryCondition, Zipper
using Oceananigans.Fields: convert_to_0_360
using Oceananigans.Grids: RightConnected
using Oceananigans.Grids: halo_size, generate_coordinate, topology
using Oceananigans.Grids: total_length, add_halos, fill_metric_halo_regions!

using Distances: haversine
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

const ZBC = BoundaryCondition{<:Zipper}

include("generate_tripolar_coordinates.jl")
include("tripolar_grid.jl")
include("tripolar_field_extensions.jl")
include("rotated_latitude_longitude_grid.jl")
include("conformal_cubed_sphere_panel.jl")

# Distributed computations on a tripolar grid
include("distributed_tripolar_grid.jl")
include("distributed_zipper.jl")
include("distributed_zipper_north_tags.jl")

# Reductions on a tripolar grid
include("tripolar_grid_reductions.jl")

end # module
