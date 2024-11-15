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


Consider the configuration

```
   Immersed      Fluid
  =========== ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅

       c           c
      i-1          i

 | ========= |           |
 × === ∘ === ×     ∘     ×
 | ========= |           |

i-1          i
 f           f           f
```

We then have

* `inactive_node(i, 1, 1, grid, f, c, c) = false`

As well as

* `inactive_node(i,   1, 1, grid, c, c, c) = false`
* `inactive_node(i-1, 1, 1, grid, c, c, c) = true`
* `inactive_node(i-1, 1, 1, grid, f, c, c) = true`
"""
@inline inactive_cell(i, j, k, ibg::IBG) = immersed_cell(i, j, k, ibg) | inactive_cell(i, j, k, ibg.underlying_grid)

# Isolate periphery of the immersed boundary
@inline immersed_peripheral_node(i, j, k, ibg::IBG, LX, LY, LZ) =  peripheral_node(i, j, k, ibg, LX, LY, LZ) &
                                                                  !peripheral_node(i, j, k, ibg.underlying_grid, LX, LY, LZ)

@inline immersed_inactive_node(i, j, k, ibg::IBG, LX, LY, LZ) =  inactive_node(i, j, k, ibg, LX, LY, LZ) &
                                                                !inactive_node(i, j, k, ibg.underlying_grid, LX, LY, LZ)

# Underlying grids are never immersed!                                                                
@inline immersed_peripheral_node(i, j, k, grid::AbstractUnderlyingGrid, LX, LY, LZ) = false
@inline   immersed_inactive_node(i, j, k, grid::AbstractUnderlyingGrid, LX, LY, LZ) = false

#####
##### Utilities
#####

const c = Center()
const f = Face()

@inline Base.zero(ibg::IBG) = zero(ibg.underlying_grid)

@inline xnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = xnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline ynode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = ynode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline znode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = znode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline λnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = λnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline φnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = φnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline ξnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = ξnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline ηnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = ηnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline rnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = rnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline node(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = node(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

nodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = nodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
nodes(ibg::IBG, (ℓx, ℓy, ℓz); kwargs...) = nodes(ibg, ℓx, ℓy, ℓz; kwargs...)

xnodes(ibg::IBG, loc; kwargs...) = xnodes(ibg.underlying_grid, loc; kwargs...)
ynodes(ibg::IBG, loc; kwargs...) = ynodes(ibg.underlying_grid, loc; kwargs...)
znodes(ibg::IBG, loc; kwargs...) = znodes(ibg.underlying_grid, loc; kwargs...)

λnodes(ibg::IBG, loc; kwargs...) = λnodes(ibg.underlying_grid, loc; kwargs...)
φnodes(ibg::IBG, loc; kwargs...) = φnodes(ibg.underlying_grid, loc; kwargs...)

ξnodes(ibg::IBG, loc; kwargs...) = ξnodes(ibg.underlying_grid, loc; kwargs...)
ηnodes(ibg::IBG, loc; kwargs...) = ηnodes(ibg.underlying_grid, loc; kwargs...)
rnodes(ibg::IBG, loc; kwargs...) = rnodes(ibg.underlying_grid, loc; kwargs...)

xnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = xnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
ynodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = ynodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
znodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = znodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)

λnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = λnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
φnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = φnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)

ξnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = ξnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
ηnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = ηnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
rnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = rnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)

@inline cpu_face_constructor_x(ibg::IBG) = cpu_face_constructor_x(ibg.underlying_grid)
@inline cpu_face_constructor_y(ibg::IBG) = cpu_face_constructor_y(ibg.underlying_grid)
@inline cpu_face_constructor_z(ibg::IBG) = cpu_face_constructor_z(ibg.underlying_grid)

node_names(ibg::IBG, ℓx, ℓy, ℓz) = node_names(ibg.underlying_grid, ℓx, ℓy, ℓz)
ξname(ibg::IBG) = ξname(ibg.underlying_grid)
ηname(ibg::IBG) = ηname(ibg.underlying_grid)
rname(ibg::IBG) = rname(ibg.underlying_grid)

function on_architecture(arch, ibg::IBG)
    underlying_grid   = on_architecture(arch, ibg.underlying_grid)
    immersed_boundary = on_architecture(arch, ibg.immersed_boundary)
    return ImmersedBoundaryGrid(underlying_grid, immersed_boundary)
end

isrectilinear(ibg::IBG) = isrectilinear(ibg.underlying_grid)

@inline fractional_x_index(x, locs, grid::ImmersedBoundaryGrid) = fractional_x_index(x, locs, grid.underlying_grid)
@inline fractional_y_index(x, locs, grid::ImmersedBoundaryGrid) = fractional_y_index(x, locs, grid.underlying_grid)
@inline fractional_z_index(x, locs, grid::ImmersedBoundaryGrid) = fractional_z_index(x, locs, grid.underlying_grid)

=======
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

# Extension to Immersed boundaries with zstar
include("abstract_zstar_immersed_grid.jl")

end # module
