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
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_uᶠᶜᶜ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_vᶜᶠᶜ, v) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_wᶜᶜᵃ, w))
end

"""
    div_xyᶜᶜᵃ(i, j, k, grid, u, v)

Calculates the 2D divergence ∂x u + ∂y v via

    1/Azᶜᶜᵃ * [δxᶜᵃᵃ(Δy * u) + δyᵃᶜᵃ(Δx * v)]

where `Azᶜᶜᵃ` is the area of the cell centered on (Center, Center, Any) --- a tracer cell,
`Δy` is the length of the cell centered on (Face, Center, Any) in `y` (a `u` cell),
and `Δx` is the length of the cell centered on (Center, Face, Any) in `x` (a `v` cell).
`div_xyᶜᶜᵃ` ends up at the location `cca`.
"""
@inline function div_xyᶜᶜᵃ(i, j, k, grid, u, v)
    return 1 / Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_uᶠᶜᶜ, u) +
                                      δyᵃᶜᵃ(i, j, k, grid, Ay_vᶜᶠᶜ, v))
end
