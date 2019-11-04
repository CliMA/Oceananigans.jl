####
#### Diffusive fluxes
####

@inline diffusive_flux_x(i, j, k, grid, c, κᶠᶜᶜ) = κᶠᶜᶜ * Axᵃᵃᶠ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, c, κᶜᶠᶜ) = κᶜᶠᶜ * Ayᵃᵃᶠ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, c, κᶜᶜᶠ) = κᶜᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, c)

####
#### Laplacian diffusion operator
####

"""
    ∇κ∇c(i, j, k, grid, κ, c)

Calculates diffusion for a tracer c via

    1/V * [δxᶜᵃᵃ(κ * Ax * δxᶠᵃᵃ(c)) + δyᵃᶜᵃ(κ * Ay * δyᵃᶠᵃ(c)) + δzᵃᵃᶜ(κ * Az * δzᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function div_κ∇c(i, j, k, grid, κ, c)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, diffusive_flux_x, c, κ) +
                                    δyᵃᶜᵃ(i, j, k, grid, diffusive_flux_y, c, κ) +
                                    δzᵃᵃᶜ(i, j, k, grid, diffusive_flux_z, c, κ))
end
