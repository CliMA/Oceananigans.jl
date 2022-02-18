module ImmersedBoundaries

export ImmerseBoundaryGrid, GridFittedBoundary, GridFittedBottom
       
using Adapt

using Oceananigans.Grids
using Oceananigans.Grids: size_summary
using Oceananigans.Operators
using Oceananigans.Fields
using Oceananigans.Utils
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, time_discretization

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
import Oceananigans.Coriolis: φᶠᶠᵃ

import Oceananigans.Grids: 
        cpu_face_constructor_x,
        cpu_face_constructor_y,
        cpu_face_constructor_z,
        on_architecture,
        architecture,
        with_halo,
        all_x_nodes,
        all_y_nodes,
        all_z_nodes,
        xnode,
        ynode,
        znode
        

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

export AbstractImmersedBoundary

"""
    abstract type AbstractImmersedBoundary

Abstract supertype for immersed boundary grids.
"""
abstract type AbstractImmersedBoundary end

struct ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    grid :: G
    immersed_boundary :: I
    
    function ImmersedBoundaryGrid{TX, TY, TZ}(grid::G, ib::I) where {TX, TY, TZ, G <: AbstractUnderlyingGrid, I}
        FT = eltype(grid)
        arch = architecture(grid)
        Arch = typeof(arch)
        return new{FT, TX, TY, TZ, G, I, Arch}(arch, grid, ib)
    end
end

function ImmersedBoundaryGrid(grid, ib)
    @warn "ImmersedBoundaryGrid is unvalidated and may produce incorrect results. " *
          "Help validate ImmersedBoundaryGrid by reporting any bugs " *
          "or unexpected behavior to https://github.com/CliMA/Oceananigans.jl/issues."
    
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib)
end

const IBG = ImmersedBoundaryGrid

@inline Base.getproperty(ibg::IBG, property::Symbol) = get_ibg_property(ibg, Val(property))
@inline get_ibg_property(ibg::IBG, ::Val{property}) where property = getfield(getfield(ibg, :grid), property)
@inline get_ibg_property(ibg::IBG, ::Val{:immersed_boundary}) = getfield(ibg, :immersed_boundary)
@inline get_ibg_property(ibg::IBG, ::Val{:grid}) = getfield(ibg, :grid)

@inline architecture(ibg::IBG) = architecture(ibg.grid)

Adapt.adapt_structure(to, ibg::IBG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    ImmersedBoundaryGrid{TX, TY, TZ}(adapt(to, ibg.grid), adapt(to, ibg.immersed_boundary))

with_halo(halo, ibg::ImmersedBoundaryGrid) = ImmersedBoundaryGrid(with_halo(halo, ibg.grid), ibg.immersed_boundary)


function Base.summary(grid::ImmersedBoundaryGrid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)

    return string(size_summary(size(grid)),
                  " ImmersedBoundaryGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo")
end

function show(io::IO, g::ImmersedBoundaryGrid)
    return print(io, "ImmersedBoundaryGrid on: \n",
                     "    architecture: $(g.architecture)\n",
                     "            grid: $(summary(g.grid))\n",
                     "   with immersed: ", typeof(g.immersed_boundary))
end

@inline cell_advection_timescale(u, v, w, ibg::ImmersedBoundaryGrid) = cell_advection_timescale(u, v, w, ibg.grid)
@inline φᶠᶠᵃ(i, j, k, ibg::ImmersedBoundaryGrid) = φᶠᶠᵃ(i, j, k, ibg.grid)

@inline xnode(LX, i, ibg::ImmersedBoundaryGrid) = xnode(LX, i, ibg.grid)
@inline ynode(LY, j, ibg::ImmersedBoundaryGrid) = ynode(LY, j, ibg.grid)
@inline znode(LZ, k, ibg::ImmersedBoundaryGrid) = znode(LZ, k, ibg.grid)

@inline xnode(LX, LY, LZ, i, j, k, ibg::ImmersedBoundaryGrid) = xnode(LX, LY, LZ, i, j, k, ibg.grid)
@inline ynode(LX, LY, LZ, i, j, k, ibg::ImmersedBoundaryGrid) = ynode(LX, LY, LZ, i, j, k, ibg.grid)
@inline znode(LX, LY, LZ, i, j, k, ibg::ImmersedBoundaryGrid) = znode(LX, LY, LZ, i, j, k, ibg.grid)

@inline cpu_face_constructor_x(ibg::ImmersedBoundaryGrid) = cpu_face_constructor_x(ibg.grid)
@inline cpu_face_constructor_y(ibg::ImmersedBoundaryGrid) = cpu_face_constructor_y(ibg.grid)
@inline cpu_face_constructor_z(ibg::ImmersedBoundaryGrid) = cpu_face_constructor_z(ibg.grid)

all_x_nodes(loc, ibg::ImmersedBoundaryGrid) = all_x_nodes(loc, ibg.grid)
all_y_nodes(loc, ibg::ImmersedBoundaryGrid) = all_y_nodes(loc, ibg.grid)
all_z_nodes(loc, ibg::ImmersedBoundaryGrid) = all_z_nodes(loc, ibg.grid)

function on_architecture(arch, ibg::ImmersedBoundaryGrid)
    underlying_grid = on_architecture(arch, ibg.grid)

    immersed_boundary = ibg.immersed_boundary isa AbstractArray ?
        arch_array(arch, ibg.immersed_boundary) :
        ibg.immersed_boundary

    return ImmersedBoundaryGrid(underlying_grid, immersed_boundary)
end

include("immersed_grid_metrics.jl")
include("grid_fitted_immersed_boundaries.jl")
include("conditional_fluxes.jl")
include("conditional_derivatives.jl")
include("mask_immersed_field.jl")
include("immersed_fields_reductions.jl")

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
            ifelse(solid_node(loc..., i, j, k, ibg), $locate_coeff(i, j, k, ibg.grid, coeff), zero(FT))
    end
end

end # module
