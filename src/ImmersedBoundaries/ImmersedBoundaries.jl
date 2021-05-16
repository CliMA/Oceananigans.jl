module ImmersedBoundaries

using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Fields
using Oceananigans.Utils
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, time_discretization

import Oceananigans.TurbulenceClosures:
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
    diffusive_flux_z,
    κᶠᶜᶜ,
    κᶜᶠᶜ,
    κᶜᶜᶠ,
    νᶜᶜᶜ,
    νᶠᶠᶜ,
    νᶜᶠᶠ,
    νᶠᶜᶠ

export AbstractImmersedBoundary

abstract type AbstractImmersedBoundary end

struct ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I} <: AbstractGrid{FT, TX, TY, TZ}
    grid :: G
    immersed_boundary :: I

    function ImmersedBoundaryGrid(grid::G, ib::I) where {G <: AbstractGrid, I}
        FT = eltype(grid)
        TX, TY, TZ = topology(grid)
        new{FT, TX, TY, TZ, G, I}(grid, ib)
    end
end

const IBG = ImmersedBoundaryGrid

@inline Base.getproperty(ibg::IBG, property::Symbol) = get_ibg_property(ibg, Val(property))
@inline get_ibg_property(ibg::IBG, ::Val{property}) where property = getfield(ibg.grid, property)
@inline get_ibg_property(ibg::IBG, ::Val{:immersed_boundary}) = getfield(ibg, :immersed_boundary)
@inline get_ibg_property(ibg::IBG, ::Val{:grid}) = getfield(ibg, :grid)

include("immersed_grid_metrics.jl")
include("grid_fitted_immersed_boundary.jl")
include("mask_immersed_field.jl")

#=
#####
##### Diffusivities (for VerticallyImplicitTimeDiscretization)
#####

for (locate_coeff, loc) in ((:κᶠᶜᶜ, (f, c, c)),
                            (:κᶜᶠᶜ, (c, f, c)),
                            (:κᶜᶜᶠ, (c, c, f)),
                            (:νᶜᶜᶜ, (c, c, c)),
                            (:νᶠᶠᶜ, (f, f, c)),
                            (:νᶠᶜᶠ, (f, c, f)),
                            (:νᶜᶠᶠ, (c, f, f)))

    @eval begin
        @inline $locate_coeff(i, j, k, grid, ib::NoImmersedBoundary, coeff) = $locate_coeff(i, j, k, grid, coeff)

        @inline $locate_coeff(i, j, k, grid::AG{FT}, ib::GFIB, coeff) where FT =
            ifelse(solid_cell(loc..., i, j, k, grid, ib), $locate_coeff(i, j, k, grid, coeff), zero(FT))
    end
end
=#

end # module
