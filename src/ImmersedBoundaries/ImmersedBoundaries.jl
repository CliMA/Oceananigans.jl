module ImmersedBoundaries

export ImmersedBoundaryGrid, GridFittedBoundary, GridFittedBottom, PartialCellBottom, ImmersedBoundaryCondition

using Adapt

using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Fields
using Oceananigans.Utils
using Oceananigans.Architectures

using Oceananigans.Grids: size_summary, inactive_node, peripheral_node, AbstractGrid

import Base: show, summary

import Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z,
                           x_domain, y_domain, z_domain

import Oceananigans.Grids: architecture, with_halo, inflate_halo_size_one_dimension,
                           xnode, ynode, znode, λnode, φnode, node,
                           ξnode, ηnode, rnode,
                           ξname, ηname, rname, node_names,
                           xnodes, ynodes, znodes, λnodes, φnodes, nodes,
                           ξnodes, ηnodes, rnodes,
                           static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ, static_column_depthᶠᶠᵃ,
                           inactive_cell


import Oceananigans.Architectures: on_architecture

import Oceananigans.Fields: fractional_x_index, fractional_y_index, fractional_z_index

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
