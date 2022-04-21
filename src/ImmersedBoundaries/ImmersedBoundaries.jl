module ImmersedBoundaries

export ImmersedBoundaryGrid, GridFittedBoundary, GridFittedBottom
       
using Adapt

using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Fields
using Oceananigans.Utils

using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, time_discretization
using Oceananigans.Grids: size_summary, inactive_node, peripheral_node

using Oceananigans.TurbulenceClosures:
    viscous_flux_ux,
    viscous_flux_uy,
    viscous_flux_uz,
    viscous_flux_vx,
    viscous_flux_vy,
    viscous_flux_vz,
    viscous_flux_wx,
    viscous_flux_wy,
    viscous_flux_wz,
    diffusive_flux_x,
    diffusive_flux_y,
    diffusive_flux_z

using Oceananigans.Advection:
    advective_momentum_flux_Uu,
    advective_momentum_flux_Uv,
    advective_momentum_flux_Uw,
    advective_momentum_flux_Vu,
    advective_momentum_flux_Vv,
    advective_momentum_flux_Vw,
    advective_momentum_flux_Wu,
    advective_momentum_flux_Wv,
    advective_momentum_flux_Ww,
    advective_tracer_flux_x,
    advective_tracer_flux_y,
    advective_tracer_flux_z,
    WENOVectorInvariant

import Base: show, summary
import Oceananigans.Utils: cell_advection_timescale
import Oceananigans.Grids: architecture, on_architecture, with_halo
import Oceananigans.Grids: xnode, ynode, znode, all_x_nodes, all_y_nodes, all_z_nodes
import Oceananigans.Grids: inactive_cell
import Oceananigans.Coriolis: φᶠᶠᵃ

import Oceananigans.Advection:
    _advective_momentum_flux_Uu,
    _advective_momentum_flux_Uv,
    _advective_momentum_flux_Uw,
    _advective_momentum_flux_Vu,
    _advective_momentum_flux_Vv,
    _advective_momentum_flux_Vw,
    _advective_momentum_flux_Wu,
    _advective_momentum_flux_Wv,
    _advective_momentum_flux_Ww,
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    _advective_tracer_flux_z

import Oceananigans.TurbulenceClosures:
    _viscous_flux_ux,
    _viscous_flux_uy,
    _viscous_flux_uz,
    _viscous_flux_vx,
    _viscous_flux_vy,
    _viscous_flux_vz,
    _viscous_flux_wx,
    _viscous_flux_wy,
    _viscous_flux_wz,
    _diffusive_flux_x,
    _diffusive_flux_y,
    _diffusive_flux_z,
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

    * `inactive_node(f, c, c, i, 1, 1, grid) = false`

As well as

    * `inactive_node(c, c, c, i,   1, 1, grid) = false`
    * `inactive_node(c, c, c, i-1, 1, 1, grid) = true`
    * `inactive_node(f, c, c, i-1, 1, 1, grid) = true`
"""
@inline inactive_cell(i, j, k, ibg::IBG) = immersed_cell(i, j, k, ibg) | inactive_cell(i, j, k, ibg.underlying_grid)

# Isolate periphery of the immersed boundary
@inline immersed_peripheral_node(LX, LY, LZ, i, j, k, ibg::IBG) =  peripheral_node(LX, LY, LZ, i, j, k, ibg) &
                                                                  !peripheral_node(LX, LY, LZ, i, j, k, ibg.underlying_grid)

#####
##### Utilities
#####

const IBG = ImmersedBoundaryGrid
const c = Center()
const f = Face()

@inline Base.zero(ibg::IBG) = zero(ibg.underlying_grid)
@inline cell_advection_timescale(u, v, w, ibg::IBG) = cell_advection_timescale(u, v, w, ibg.underlying_grid)
@inline φᶠᶠᵃ(i, j, k, ibg::IBG) = φᶠᶠᵃ(i, j, k, ibg.underlying_grid)

@inline xnode(LX, i, ibg::IBG) = xnode(LX, i, ibg.underlying_grid)
@inline ynode(LY, j, ibg::IBG) = ynode(LY, j, ibg.underlying_grid)
@inline znode(LZ, k, ibg::IBG) = znode(LZ, k, ibg.underlying_grid)

@inline xnode(LX, LY, LZ, i, j, k, ibg::IBG) = xnode(LX, LY, LZ, i, j, k, ibg.underlying_grid)
@inline ynode(LX, LY, LZ, i, j, k, ibg::IBG) = ynode(LX, LY, LZ, i, j, k, ibg.underlying_grid)
@inline znode(LX, LY, LZ, i, j, k, ibg::IBG) = znode(LX, LY, LZ, i, j, k, ibg.underlying_grid)

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
include("conditional_derivatives.jl")
include("mask_immersed_field.jl")
include("immersed_reductions.jl")

end # module
