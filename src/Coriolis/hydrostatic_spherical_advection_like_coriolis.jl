# Our two Coriolis schemes are energy-conserving or enstrophy-conserving
# with a "vector invariant" momentum advection scheme, but not with a "flux form"
# or "conservation form" advection scheme (which does not currently exist for
# curvilinear grids).

using Oceananigans.Advection: AbstractAdvectionScheme, _left_biased_interpolate_yᵃᶜᵃ, _right_biased_interpolate_xᶜᵃᵃ
"""
    struct HydrostaticSphericalCoriolis{S, FT} <: AbstractRotation

A parameter object for constant rotation around a vertical axis.
"""

const AdvectionLikeCoriolis = HydrostaticSphericalCoriolis{<:AbstractAdvectionScheme}

######
###### Horizontally-vector-invariant formulation of momentum scheme
######
###### Follows https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#vector-invariant-momentum-equations
######

@inline function x_f_cross_U(i, j, k, grid, coriolis::AdvectionLikeCoriolis, U)
    v̂  =  ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, U.v) / Δxᶠᶜᶜ(i, j, k, grid) 
    fᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, coriolis.scheme, fᶠᶠᵃ, coriolis)
    fᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, coriolis.scheme, fᶠᶠᵃ, coriolis)
    return - upwind_biased_product(v̂, fᴸ, fᴿ) 
end

@inline function y_f_cross_U(i, j, k, grid, coriolis::AdvectionLikeCoriolis, U) 
    û  =  ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, U.u) / Δyᶜᶠᶜ(i, j, k, grid)
    fᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, coriolis.scheme, fᶠᶠᵃ, coriolis)
    fᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, coriolis.scheme, fᶠᶠᵃ, coriolis)
    return + upwind_biased_product(û, fᴸ, fᴿ) 
end
