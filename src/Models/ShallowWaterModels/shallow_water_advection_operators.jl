using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: Ax_ψᵃᵃᶜ, Ay_ψᵃᵃᶜ

#####
##### Momentum flux operators
#####

@inline momentum_flux_huu(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_uu(i, j, k, grid, advection, solution.uh, solution.uh) / solution.h[i, j, k]

@inline momentum_flux_huv(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_uv(i, j, k, grid, advection, solution.uh, solution.vh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline momentum_flux_hvu(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_vu(i, j, k, grid, advection, solution.uh, solution.vh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline momentum_flux_hvv(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_vv(i, j, k, grid, advection, solution.vh, solution.vh) / solution.h[i, j, k]

#####
##### Momentum flux divergence operators
#####

@inline div_hUu(i, j, k, grid, advection, solution) =
    1 / Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_huu, advection, solution) +
                               δyᵃᶜᵃ(i, j, k, grid, momentum_flux_huv, advection, solution))

@inline div_hUv(i, j, k, grid, advection, solution) =
    1 / Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_hvu, advection, solution) +
                               δyᵃᶠᵃ(i, j, k, grid, momentum_flux_hvv, advection, solution))

# Support for no advection
@inline div_hUu(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution) where FT = zero(FT)
@inline div_hUv(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution) where FT = zero(FT)

#####
##### Mass transport divergence operator
#####

"""
    div_uhvh(i, j, k, grid, solution)

Calculates the divergence of the mass flux into a cell,

    1/V * [δxᶜᵃᵃ(Ax * uh) + δyᵃᶜᵃ(Ay * vh)]

which will end up at the location `ccc`.
"""
@inline function div_uhvh(i, j, k, grid, solution)
    1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶜ, solution.uh) + 
                             δyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶜ, solution.vh))
end
