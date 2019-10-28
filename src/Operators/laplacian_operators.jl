"""
    ∇²(i, j, k, grid, c)

Calculates the Laplacian of c via

    1/V * [δxᶜᵃᵃ(Ax * δxᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * δyᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * δzᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function ∇²(i, j, k, grid, c)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, δᴶxᶠᵃᵃ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, δᴶyᵃᶠᵃ, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, δᴶzᵃᵃᶠ, c))
end

