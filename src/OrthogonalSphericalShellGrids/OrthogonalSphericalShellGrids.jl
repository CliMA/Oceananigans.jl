module OrthogonalSphericalShellGrids

# The only thing we need!
export TripolarGrid, RotatedLatitudeLongitudeGrid, ConformalCubedSpherePanelGrid

import Oceananigans
import Oceananigans.Architectures: on_architecture

using Oceananigans.Architectures: on_architecture, AbstractArchitecture, CPU, GPU
using Oceananigans.BoundaryConditions: BoundaryCondition
using Oceananigans.Grids: AbstractTopology
using Oceananigans.Grids: halo_size, generate_coordinate, topology
using Oceananigans.Grids: total_length, add_halos, fill_metric_halo_regions!
using Oceananigans.BoundaryConditions: fill_halo_regions!

using Distances: haversine
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

include("generate_tripolar_coordinates.jl")
include("tripolar_grid.jl")
include("tripolar_field_extensions.jl")
include("right_face_folded_kernel_parameters.jl")
include("rotated_latitude_longitude_grid.jl")
include("conformal_cubed_sphere_panel.jl")

# Distributed computations on a tripolar grid
include("distributed_tripolar_grid.jl")
include("distributed_zipper.jl")
include("distributed_zipper_north_tags.jl")

# Fallback for OSSG variants without a tailored constructor_arguments method
# (e.g. ConformalCubedSpherePanelGrid). Returns minimal info so that NetCDFWriter
# can still write data — reconstruction of these grids from the file is not yet
# implemented and will raise an informative error in reconstruct_grid.
using OrderedCollections: OrderedDict
using Oceananigans.Grids: Grids
function Grids.constructor_arguments(grid::Oceananigans.Grids.OrthogonalSphericalShellGrid)
    args = OrderedDict{Symbol, Any}(:architecture => Oceananigans.Grids.architecture(grid),
                                    :number_type  => eltype(grid))
    kwargs = Dict{Symbol, Any}(:size                       => size(grid),
                               :halo                       => (grid.Hx, grid.Hy, grid.Hz),
                               :reconstruction_unsupported => true)
    return args, kwargs
end

end # module
