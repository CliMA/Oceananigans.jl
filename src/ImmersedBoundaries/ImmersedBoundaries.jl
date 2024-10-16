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

import Oceananigans.Grids: architecture, on_architecture, with_halo, inflate_halo_size_one_dimension,
                           xnode, ynode, znode, λnode, φnode, node,
                           ξnode, ηnode, rnode,
                           ξname, ηname, rname, node_names,
                           xnodes, ynodes, znodes, λnodes, φnodes, nodes,
                           ξnodes, ηnodes, rnodes,
                           inactive_cell


import Oceananigans.Fields: fractional_x_index, fractional_y_index, fractional_z_index

"""
    abstract type AbstractImmersedBoundary

Abstract supertype for immersed boundary grids.
"""
abstract type AbstractImmersedBoundary end

#####
##### ImmersedBoundaryGrid
#####

struct ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    underlying_grid :: G
    immersed_boundary :: I
    interior_active_cells :: M
    active_z_columns :: S

    # Internal interface
    function ImmersedBoundaryGrid{TX, TY, TZ}(grid::G, ib::I, mi::M, ms::S) where {TX, TY, TZ, G <: AbstractUnderlyingGrid, I, M, S}
        FT = eltype(grid)
        arch = architecture(grid)
        Arch = typeof(arch)
        return new{FT, TX, TY, TZ, G, I, M, S, Arch}(arch, grid, ib, mi, ms)
    end

    # Constructor with no active map
    function ImmersedBoundaryGrid{TX, TY, TZ}(grid::G, ib::I) where {TX, TY, TZ, G <: AbstractUnderlyingGrid, I}
        FT = eltype(grid)
        arch = architecture(grid)
        Arch = typeof(arch)
        return new{FT, TX, TY, TZ, G, I, Nothing, Nothing, Arch}(arch, grid, ib, nothing, nothing)
    end
end

const IBG = ImmersedBoundaryGrid

@inline Base.getproperty(ibg::IBG, property::Symbol) = get_ibg_property(ibg, Val(property))
@inline get_ibg_property(ibg::IBG, ::Val{property}) where property = getfield(getfield(ibg, :underlying_grid), property)
@inline get_ibg_property(ibg::IBG, ::Val{:immersed_boundary})      = getfield(ibg, :immersed_boundary)
@inline get_ibg_property(ibg::IBG, ::Val{:underlying_grid})        = getfield(ibg, :underlying_grid)
@inline get_ibg_property(ibg::IBG, ::Val{:interior_active_cells})  = getfield(ibg, :interior_active_cells)
@inline get_ibg_property(ibg::IBG, ::Val{:active_z_columns})       = getfield(ibg, :active_z_columns)

@inline architecture(ibg::IBG) = architecture(ibg.underlying_grid)

@inline x_domain(ibg::IBG) = x_domain(ibg.underlying_grid)
@inline y_domain(ibg::IBG) = y_domain(ibg.underlying_grid)
@inline z_domain(ibg::IBG) = z_domain(ibg.underlying_grid)

Adapt.adapt_structure(to, ibg::IBG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    ImmersedBoundaryGrid{TX, TY, TZ}(adapt(to, ibg.underlying_grid),
                                     adapt(to, ibg.immersed_boundary),
                                     nothing,
                                     nothing)

with_halo(halo, ibg::ImmersedBoundaryGrid) =
    ImmersedBoundaryGrid(with_halo(halo, ibg.underlying_grid), ibg.immersed_boundary)

# ImmersedBoundaryGrids require an extra halo point to check the "inactivity" of a `Face` node at N + H
# (which requires checking `Center` nodes at N + H and N + H + 1)
inflate_halo_size_one_dimension(req_H, old_H, _, ::IBG)            = max(req_H + 1, old_H)
inflate_halo_size_one_dimension(req_H, old_H, ::Type{Flat}, ::IBG) = 0

# Defining the bottom
@inline z_bottom(i, j, grid) = znode(i, j, 1, grid, c, c, f)
@inline z_bottom(i, j, ibg::IBG) = error("The function `bottom` has not been defined for $(summary(ibg))!")

function Base.summary(grid::ImmersedBoundaryGrid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)

    return string(size_summary(size(grid)),
                  " ImmersedBoundaryGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo")
end

function show(io::IO, ibg::ImmersedBoundaryGrid)
    print(io, summary(ibg), ":", "\n",
             "├── immersed_boundary: ", summary(ibg.immersed_boundary), "\n",
             "├── underlying_grid: ", summary(ibg.underlying_grid), "\n")

    return show(io, ibg.underlying_grid, false)
end

#####
##### Interface for immersed_boundary
#####

"""
    immersed_cell(i, j, k, grid)

Return true if a `cell` is "completely" immersed, and thus
is not part of the prognostic state.
"""
@inline immersed_cell(i, j, k, grid) = false

# Unpack to make defining new immersed boundaries more convenient
@inline immersed_cell(i, j, k, grid::ImmersedBoundaryGrid) =
    immersed_cell(i, j, k, grid.underlying_grid, grid.immersed_boundary)

"""
    inactive_cell(i, j, k, grid::ImmersedBoundaryGrid)

Return `true` if the tracer cell at `i, j, k` either (i) lies outside the `Bounded` domain
or (ii) lies within the immersed region of `ImmersedBoundaryGrid`.

Example
=======

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

end # module
