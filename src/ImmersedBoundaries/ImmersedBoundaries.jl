module ImmersedBoundaries

export ImmersedBoundaryGrid, GridFittedBoundary, GridFittedBottom, PartialCellBottom,
       GridFittedVerticalBoundaries, PartialCellVerticalBoundaries, ImmersedBoundaryCondition,
       bottom_height_field, materialize_immersed_boundary

using Printf: @sprintf

using Oceananigans.Architectures: Architectures, on_architecture
using Oceananigans.Grids: Grids, AbstractGrid, Center, Face, Flat, architecture, cpu_face_constructor_x,
                          cpu_face_constructor_y, cpu_face_constructor_z, immersed_inactive_node,
                          immersed_peripheral_node, inactive_cell, inactive_node, node, node_names, nodes,
                          peripheral_node, rname, rnode, rnodes, size_summary, static_column_depthᶜᶜᵃ,
                          static_column_depthᶜᶠᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶠᶠᵃ, with_halo, x_domain,
                          xnode, xnodes, y_domain, ynode, ynodes, z_domain, znode, znodes, ηname, ηnode, ηnodes, λnode,
                          λnodes, ξname, ξnode, ξnodes, φnode, φnodes
using Oceananigans.Utils: @apply_regionally, launch!

using Adapt: Adapt, adapt

using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans.AbstractOperations: AbstractOperations
using Oceananigans.BoundaryConditions: BoundaryConditions
using Oceananigans.Fields: Fields, AbstractField, ConstantField, fractional_x_index, fractional_y_index,
                           fractional_z_index
using Oceananigans.Operators: Operators

include("immersed_boundary_grid.jl")
include("immersed_boundary_interface.jl")
include("immersed_boundary_nodes.jl")
include("active_cells_map.jl")
include("immersed_grid_metrics.jl")
include("abstract_grid_fitted_boundary.jl")
include("grid_fitted_boundary.jl")
include("grid_fitted_bottom.jl")
include("partial_cell_bottom.jl")
include("immersed_boundary_condition.jl")
include("conditional_differences.jl")
include("mask_immersed_field.jl")
include("immersed_reductions.jl")
include("mutable_immersed_grid.jl")

end # module
