"""
    ∇²(i, j, k, grid, c)

Calculates the Laplacian of c via

    1/V * [δxᶜᵃᵃ(Ax * ∂xᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * ∂yᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * ∂zᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function ∇²(i, j, k, grid, c)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᵃᵃ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᵃᶠᵃ, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_∂zᵃᵃᶠ, c))
end
