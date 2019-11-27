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
    ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, κˣ, κʸ, κᶻ, c)

Calculates diffusion for a tracer c via

    1/V * [δxᶜᵃᵃ(κˣ * Ax * ∂xᶠᵃᵃ(c)) + δyᵃᶜᵃ(κʸ * Ay * ∂yᵃᶠᵃ(c)) + δzᵃᵃᶜ(κᶻ * Az * ∂zᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, κˣ, κʸ, κᶻ, c)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, diffusive_flux_x, κˣ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, diffusive_flux_y, κʸ, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, diffusive_flux_z, κᶻ, c))
end

@inline ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, κ, c) = ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, κ, κ, κ, c)
