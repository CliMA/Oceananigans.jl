module ImmersedBoundaries

export ImmersedBoundaryGrid, GridFittedBoundary, GridFittedBottom, ImmersedBoundaryCondition
       
using Adapt

using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Fields
using Oceananigans.Utils

using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, time_discretization
using Oceananigans.Grids: size_summary, inactive_node, peripheral_node, x_boundary, y_boundary, z_boundary
using Oceananigans.Advection: WENOVectorInvariant

import Base: show, summary
import Oceananigans.Utils: cell_advection_timescale
import Oceananigans.Grids: architecture, on_architecture, with_halo
import Oceananigans.Grids: xnode, ynode, znode, all_x_nodes, all_y_nodes, all_z_nodes
import Oceananigans.Grids: inactive_cell
import Oceananigans.Coriolis: φᶠᶠᵃ

import Oceananigans.TurbulenceClosures:
    κᶠᶜᶜ,
    κᶜᶠᶜ,
    κᶜᶜᶠ,
    νᶜᶜᶜ,
    νᶠᶠᶜ,
    νᶜᶠᶠ,
    νᶠᶜᶠ

"""
    abstract type AbstractImmersedBoundary

Abstract supertype for immersed boundary grids.
"""
abstract type AbstractImmersedBoundary end

#####
##### ImmersedBoundaryGrid
#####

struct ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    underlying_grid :: G
    immersed_boundary :: I
    
    # Internal interface
    function ImmersedBoundaryGrid{TX, TY, TZ}(grid::G, ib::I) where {TX, TY, TZ, G <: AbstractUnderlyingGrid, I}
        FT = eltype(grid)
        arch = architecture(grid)
        Arch = typeof(arch)
        return new{FT, TX, TY, TZ, G, I, Arch}(arch, grid, ib)
    end
end

const IBG = ImmersedBoundaryGrid

@inline Base.getproperty(ibg::IBG, property::Symbol) = get_ibg_property(ibg, Val(property))
@inline get_ibg_property(ibg::IBG, ::Val{property}) where property = getfield(getfield(ibg, :underlying_grid), property)
@inline get_ibg_property(ibg::IBG, ::Val{:immersed_boundary}) = getfield(ibg, :immersed_boundary)
@inline get_ibg_property(ibg::IBG, ::Val{:underlying_grid}) = getfield(ibg, :underlying_grid)

@inline architecture(ibg::IBG) = architecture(ibg.underlying_grid)

Adapt.adapt_structure(to, ibg::IBG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    ImmersedBoundaryGrid{TX, TY, TZ}(adapt(to, ibg.underlying_grid), adapt(to, ibg.immersed_boundary))

with_halo(halo, ibg::ImmersedBoundaryGrid) = ImmersedBoundaryGrid(with_halo(halo, ibg.underlying_grid), ibg.immersed_boundary)

function Base.summary(grid::ImmersedBoundaryGrid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)

    return string(size_summary(size(grid)),
                  " ImmersedBoundaryGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo")
end

function show(io::IO, ibg::ImmersedBoundaryGrid)
    print(io, summary(ibg), ":", '\n',
              "├── immersed_boundary: ", summary(ibg.immersed_boundary), '\n',
              "├── underlying_grid: ", summary(ibg.underlying_grid), '\n')

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
@inline immersed_peripheral_node(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) =  peripheral_node(i, j, k, ibg, ℓx, ℓy, ℓz) &
                                                                  !peripheral_node(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

# Isolate periphery of the immersed boundary
@inline immersed_x_boundary(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = x_boundary(i, j, k, ibg, ℓx, ℓy, ℓz) & !x_boundary(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline immersed_y_boundary(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = y_boundary(i, j, k, ibg, ℓx, ℓy, ℓz) & !y_boundary(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline immersed_z_boundary(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = z_boundary(i, j, k, ibg, ℓx, ℓy, ℓz) & !z_boundary(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

#####
##### Utilities
#####

const IBG = ImmersedBoundaryGrid
const c = Center()
const f = Face()

@inline Base.zero(ibg::IBG) = zero(ibg.underlying_grid)
@inline cell_advection_timescale(u, v, w, ibg::IBG) = cell_advection_timescale(u, v, w, ibg.underlying_grid)
@inline φᶠᶠᵃ(i, j, k, ibg::IBG) = φᶠᶠᵃ(i, j, k, ibg.underlying_grid)

@inline xnode(ℓx, i, ibg::IBG) = xnode(ℓx, i, ibg.underlying_grid)
@inline ynode(ℓy, j, ibg::IBG) = ynode(ℓy, j, ibg.underlying_grid)
@inline znode(ℓz, k, ibg::IBG) = znode(ℓz, k, ibg.underlying_grid)

@inline xnode(ℓx, ℓy, ℓz, i, j, k, ibg::IBG) = xnode(ℓx, ℓy, ℓz, i, j, k, ibg.underlying_grid)
@inline ynode(ℓx, ℓy, ℓz, i, j, k, ibg::IBG) = ynode(ℓx, ℓy, ℓz, i, j, k, ibg.underlying_grid)
@inline znode(ℓx, ℓy, ℓz, i, j, k, ibg::IBG) = znode(ℓx, ℓy, ℓz, i, j, k, ibg.underlying_grid)

all_x_nodes(loc, ibg::IBG) = all_x_nodes(loc, ibg.underlying_grid)
all_y_nodes(loc, ibg::IBG) = all_y_nodes(loc, ibg.underlying_grid)
all_z_nodes(loc, ibg::IBG) = all_z_nodes(loc, ibg.underlying_grid)

function on_architecture(arch, ibg::IBG)
    underlying_grid   = on_architecture(arch, ibg.underlying_grid)
    immersed_boundary = on_architecture(arch, ibg.immersed_boundary)
    return ImmersedBoundaryGrid(underlying_grid, immersed_boundary)
end

isrectilinear(ibg::IBG) = isrectilinear(ibg.underlying_grid)

#####
##### Diffusivities (for VerticallyImplicit)
##### (the diffusivities on the immersed boundaries are kept)
#####

for (locate_coeff, loc) in ((:κᶠᶜᶜ, (f, c, c)),
                            (:κᶜᶠᶜ, (c, f, c)),
                            (:κᶜᶜᶠ, (c, c, f)),
                            (:νᶜᶜᶜ, (c, c, c)),
                            (:νᶠᶠᶜ, (f, f, c)),
                            (:νᶠᶜᶠ, (f, c, f)),
                            (:νᶜᶠᶠ, (c, f, f)))

    @eval begin
        @inline $locate_coeff(i, j, k, ibg::IBG{FT}, coeff) where FT =
            ifelse(inactive_node(loc..., i, j, k, ibg), $locate_coeff(i, j, k, ibg.underlying_grid, coeff), zero(FT))
    end
end

include("immersed_grid_metrics.jl")
include("grid_fitted_immersed_boundaries.jl")
include("conditional_fluxes.jl")
include("immersed_boundary_condition.jl")
include("mask_immersed_field.jl")
include("immersed_reductions.jl")

end # module
