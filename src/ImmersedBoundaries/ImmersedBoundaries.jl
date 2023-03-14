module ImmersedBoundaries

export ImmersedBoundaryGrid, GridFittedBoundary, GridFittedBottom, ImmersedBoundaryCondition
       
using Adapt

using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Fields
using Oceananigans.Utils
using Oceananigans.Architectures

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
    advective_tracer_flux_z
    
import Base: show, summary
import Oceananigans.Utils: cell_advection_timescale

import Oceananigans.Grids: 
        cpu_face_constructor_x,
        cpu_face_constructor_y,
        cpu_face_constructor_z,
        x_domain,
        y_domain,
        z_domain
        
import Oceananigans.Grids: architecture, on_architecture, with_halo, inflate_halo_size_one_dimension
import Oceananigans.Grids: xnode, ynode, znode, xnodes, ynodes, znodes
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
    νᶠᶜᶠ,
    z_bottom

"""
    abstract type AbstractImmersedBoundary

Abstract supertype for immersed boundary grids.
"""
abstract type AbstractImmersedBoundary end

#####
##### ImmersedBoundaryGrid
#####

struct ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    underlying_grid :: G
    immersed_boundary :: I
    active_cells_map :: M
    
    # Internal interface
    function ImmersedBoundaryGrid{TX, TY, TZ}(grid::G, ib::I, wcm::M) where {TX, TY, TZ, G <: AbstractUnderlyingGrid, I, M}
        FT = eltype(grid)
        arch = architecture(grid)
        Arch = typeof(arch)
        
        return new{FT, TX, TY, TZ, G, I, M, Arch}(arch, grid, ib, wcm)
    end
end

const IBG = ImmersedBoundaryGrid

@inline Base.getproperty(ibg::IBG, property::Symbol) = get_ibg_property(ibg, Val(property))
@inline get_ibg_property(ibg::IBG, ::Val{property}) where property = getfield(getfield(ibg, :underlying_grid), property)
@inline get_ibg_property(ibg::IBG, ::Val{:immersed_boundary})  = getfield(ibg, :immersed_boundary)
@inline get_ibg_property(ibg::IBG, ::Val{:underlying_grid})    = getfield(ibg, :underlying_grid)
@inline get_ibg_property(ibg::IBG, ::Val{:active_cells_map})   = getfield(ibg, :active_cells_map)

@inline architecture(ibg::IBG) = architecture(ibg.underlying_grid)

@inline x_domain(ibg::IBG) = x_domain(ibg.underlying_grid)
@inline y_domain(ibg::IBG) = y_domain(ibg.underlying_grid)
@inline z_domain(ibg::IBG) = z_domain(ibg.underlying_grid)

Adapt.adapt_structure(to, ibg::IBG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    ImmersedBoundaryGrid{TX, TY, TZ}(adapt(to, ibg.underlying_grid), adapt(to, ibg.immersed_boundary), adapt(to, ibg.active_cells_map))

with_halo(halo, ibg::ImmersedBoundaryGrid) = ImmersedBoundaryGrid(with_halo(halo, ibg.underlying_grid), ibg.immersed_boundary)

# ImmersedBoundaryGrids require an extra halo point to check the "inactivity" of a `Face` node at N + H 
# (which requires checking `Center` nodes at N + H and N + H + 1)
inflate_halo_size_one_dimension(req_H, old_H, _, ::IBG)            = max(req_H + 1, old_H)
inflate_halo_size_one_dimension(req_H, old_H, ::Type{Flat}, ::IBG) = 0

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
@inline cell_advection_timescale(u, v, w, ibg::IBG) = cell_advection_timescale(u, v, w, ibg.underlying_grid)
@inline φᶠᶠᵃ(i, j, k, ibg::IBG) = φᶠᶠᵃ(i, j, k, ibg.underlying_grid)

@inline xnode(i, ibg::IBG, LX; kwargs...) = xnode(i, ibg.underlying_grid, LX; kwargs...)
@inline ynode(j, ibg::IBG, LY; kwargs...) = ynode(j, ibg.underlying_grid, LY; kwargs...)
@inline znode(k, ibg::IBG, LZ; kwargs...) = znode(k, ibg.underlying_grid, LZ; kwargs...)

@inline xnode(i, j, k, ibg::IBG, LX, LY, LZ; kwargs...) = xnode(i, j, k, ibg.underlying_grid, LX, LY, LZ; kwargs...)
@inline ynode(i, j, k, ibg::IBG, LX, LY, LZ; kwargs...) = ynode(i, j, k, ibg.underlying_grid, LX, LY, LZ; kwargs...)
@inline znode(i, j, k, ibg::IBG, LX, LY, LZ; kwargs...) = znode(i, j, k, ibg.underlying_grid, LX, LY, LZ; kwargs...)

xnodes(ibg::IBG, loc; kwargs...) = xnodes(ibg.underlying_grid, loc; kwargs...)
ynodes(ibg::IBG, loc; kwargs...) = ynodes(ibg.underlying_grid, loc; kwargs...)
znodes(ibg::IBG, loc; kwargs...) = znodes(ibg.underlying_grid, loc; kwargs...)

@inline cpu_face_constructor_x(ibg::IBG) = cpu_face_constructor_x(ibg.underlying_grid)
@inline cpu_face_constructor_y(ibg::IBG) = cpu_face_constructor_y(ibg.underlying_grid)
@inline cpu_face_constructor_z(ibg::IBG) = cpu_face_constructor_z(ibg.underlying_grid)

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
            ifelse(inactive_node(i, j, k, ibg, loc...), $locate_coeff(i, j, k, ibg.underlying_grid, coeff), zero(FT))
    end
end

include("active_cells_map.jl")
include("immersed_grid_metrics.jl")
include("grid_fitted_immersed_boundaries.jl")
include("partial_cell_immersed_boundaries.jl")
include("conditional_fluxes.jl")
include("immersed_boundary_condition.jl")
include("conditional_derivatives.jl")
include("mask_immersed_field.jl")
include("immersed_reductions.jl")

end # module
