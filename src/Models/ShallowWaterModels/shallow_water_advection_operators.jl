using Oceananigans.Grids: AbstractGrid

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
##### Mass flux and flux divergence operators
#####

@inline mass_flux_x(i, j, k, grid, uh) = @inbounds Ax_ψᵃᵃᶠ(i, j, k, grid, uh)
@inline mass_flux_y(i, j, k, grid, vh) = @inbounds Ay_ψᵃᵃᶠ(i, j, k, grid, vh) 

div_UV(i, j, k, grid, advection, solution) =
    1 / Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, mass_flux_x, advection, solution.uh) +
                               δyᵃᶜᵃ(i, j, k, grid, mass_flux_y, advection, solution.vh))
