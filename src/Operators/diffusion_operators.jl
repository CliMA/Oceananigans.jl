####
#### Diffusive fluxes
####

@inline diffusive_flux_x(i, j, k, grid, κᶠᶜᶜ, c) = κᶠᶜᶜ * Axᵃᵃᶠ(i, j, k, grid) * ∂xᶠᵃᵃ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, κᶜᶠᶜ, c) = κᶜᶠᶜ * Ayᵃᵃᶠ(i, j, k, grid) * ∂yᵃᶠᵃ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, κᶜᶜᶠ, c) = κᶜᶜᶠ * Azᵃᵃᵃ(i, j, k, grid) * ∂zᵃᵃᶠ(i, j, k, grid, c)

####
#### Laplacian diffusion operator
####

"""
    ∇κ∇c(i, j, k, grid, κ, c)

Calculates diffusion for a tracer c via

    1/V * [δxᶜᵃᵃ(κ * Ax * ∂xᶠᵃᵃ(c)) + δyᵃᶜᵃ(κ * Ay * ∂yᵃᶠᵃ(c)) + δzᵃᵃᶜ(κ * Az * ∂zᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function div_κ∇c(i, j, k, grid, κ, c)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, diffusive_flux_x, κ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, diffusive_flux_y, κ, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, diffusive_flux_z, κ, c))
end
