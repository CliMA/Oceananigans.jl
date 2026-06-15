module OrthogonalSphericalShellGrids

# The only thing we need!
export TripolarGrid, RotatedLatitudeLongitudeGrid, ConformalCubedSpherePanelGrid

import Oceananigans
import Oceananigans.Architectures: on_architecture

using Distances: haversine
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans.Architectures: on_architecture, AbstractArchitecture, CPU, GPU
using Oceananigans.BoundaryConditions: BoundaryCondition
using Oceananigans.Grids: AbstractTopology
using Oceananigans.Grids: halo_size, generate_coordinate, topology
using Oceananigans.Grids: total_length, add_halos, fill_metric_halo_regions!
using Oceananigans.BoundaryConditions: fill_halo_regions!

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

# `constructor_arguments` for an `OrthogonalSphericalShellGrid`. OSSG itself is a
# general orthogonal grid on the surface of a sphere; `TripolarGrid`,
# `RotatedLatitudeLongitudeGrid`, and `ConformalCubedSpherePanelGrid` are type
# aliases of OSSG parameterized by different `conformal_mapping`s — they're
# different *ways to generate* an OSSG, not different grid types. NetCDFWriter
# does not replay those generator functions on read: it rebuilds the OSSG
# directly from the metric/coordinate arrays written to disk (see
# `reconstruct_ossg_grid` in the NetCDF extension). The metadata here is just
# what the reconstruction routine needs that isn't already on the metric arrays
# themselves: architecture, FT, size, halo, topology, radius.
using OrderedCollections: OrderedDict
using Oceananigans.Grids: Grids
function Grids.constructor_arguments(grid::Oceananigans.Grids.OrthogonalSphericalShellGrid)
    args = OrderedDict{Symbol, Any}(:architecture => Oceananigans.Grids.architecture(grid),
                                    :number_type  => eltype(grid))
    kwargs = Dict{Symbol, Any}(:size     => size(grid),
                               :halo     => (grid.Hx, grid.Hy, grid.Hz),
                               :topology => Oceananigans.Grids.topology(grid),
                               :radius   => grid.radius)
    return args, kwargs
end

# Serialized description of `OrthogonalSphericalShellGrid.conformal_mapping`. The full
# struct can't be a NetCDF attribute (no nesting), so it's flattened to a `Dict` whose
# entries are themselves NetCDF-attribute-friendly scalars/tuples. `reconstruct_ossg_grid`
# uses this dict to rebuild the conformal mapping with the right type — restoring the
# `TripolarGrid` / `RotatedLatitudeLongitudeGrid` type-alias on read.

"""
$(TYPEDSIGNATURES)

Return a `Dict{Symbol, Any}` describing the conformal mapping `cm` (or `nothing`) of an
`OrthogonalSphericalShellGrid`, suitable for serialization to NetCDF attributes.
"""
conformal_mapping_info(::Nothing) = Dict{Symbol, Any}(:type => "Nothing")

conformal_mapping_info(cm::Tripolar) = Dict{Symbol, Any}(
    :type                  => "Tripolar",
    :north_poles_latitude  => cm.north_poles_latitude,
    :first_pole_longitude  => cm.first_pole_longitude,
    :southernmost_latitude => cm.southernmost_latitude,
)

conformal_mapping_info(cm::LatitudeLongitudeRotation) = Dict{Symbol, Any}(
    :type        => "LatitudeLongitudeRotation",
    :north_pole_λ => cm.north_pole[1],
    :north_pole_φ => cm.north_pole[2],
)

# Fallback for conformal mappings without an explicit `conformal_mapping_info` method
# (e.g. `CubedSphereConformalMapping`, whose ξ/η arrays and rotation don't fit cleanly
# as NetCDF attributes). We record only the type name; the reconstructed OSSG will have
# `conformal_mapping = nothing`.
#
# This is a metadata-only loss. The grid itself reconstructs faithfully from the saved
# metric/coordinate arrays and is fully usable as a generic `OrthogonalSphericalShellGrid`
# — `conformal_mapping` is bookkeeping that names *how* the grid was originally generated,
# not data the grid uses at runtime. The only consequence is that the type-alias identity
# (e.g. `ConformalCubedSpherePanelGrid`) is not restored on this code path.
conformal_mapping_info(cm) = Dict{Symbol, Any}(:type => string(typeof(cm).name.wrapper))

end # module
