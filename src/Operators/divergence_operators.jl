#####
##### Divergence operators
#####

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

"""
    div_xyᶜᶜᵃ(i, j, k, grid, u, v)

Calculates the 2D divergence ∂x u + ∂y v via

    1/V * [δxᶜᵃᵃ(Ax * u) + δyᵃᶜᵃ(Ay * v)]

which will end up at the location `cca`.
"""
@inline function div_xyᶜᶜᵃ(i, j, k, grid, u, v)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v))
end

"""
    div_xzᶜᶜᵃ(i, j, k, grid, u, w)

Calculates the 2D divergence ∂x u + ∂z w via

    1/V * [δxᶜᵃᵃ(Ax * u) + δzᵃᵃᶜ(Az * w)]

which will end up at the location `cac`.
"""
@inline function div_xzᶜᵃᶜ(i, j, k, grid, u, w)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, w))
end

"""
    div_yzᶜᶜᵃ(i, j, k, grid, v, w)

Calculates the 2D divergence ∂y v + ∂z w via

    1/V * [δyᶜᵃᵃ(Ay * v) + δzᵃᵃᶜ(Az * w)]

which will end up at the location `acc`.
"""
@inline function div_yzᵃᶜᶜ(i, j, k, grid, v, w)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, w))
end
