#####
#####Divergence operators
#####

"""
    hdivᶜᶜᵃ(i, j, k, grid, u, v)

Calculates the horizontal divergence ∇ₕ·(u, v) of a 2D velocity field (u, v) via

    1/V * [δxᶜᵃᵃ(Ax * u) + δyᵃᶜᵃ(Ay * v)]

which will end up at the location `cca`.
"""
@inline function hdivᶜᶜᵃ(i, j, k, grid, u, v)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v))
end

"""
    divᶜᶜᶜ(i, j, k, grid, u, v, w)

Calculates the divergence ∇·U of a vector field U = (u, v, w),

    1/V * [δxᶜᵃᵃ(Ax * u) + δxᵃᶜᵃ(Ay * v) + δzᵃᵃᶜ(Az * w)],

which will end up at the cell centers `ccc`.
"""
@inline function divᶜᶜᶜ(i, j, k, grid, u, v, w)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, w))
end
