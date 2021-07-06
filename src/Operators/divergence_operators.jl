#####
##### Divergence operators
#####

"""
    divᶜᶜᶜ(i, j, k, grid, u, v, w)

Calculates the divergence ∇·𝐔 of a vector field 𝐔 = (u, v, w),

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

Returns the discrete `div_xy = ∂x u + ∂y v` of velocity field `u, v` defined as

```
1 / Azᶜᶜᵃ * [δxᶜᵃᵃ(Δy * u) + δyᵃᶜᵃ(Δx * v)]
```

at `i, j, k`, where `Azᶜᶜᵃ` is the area of the cell centered on (Center, Center, Any) --- a tracer cell,
`Δy` is the length of the cell centered on (Face, Center, Any) in `y` (a `u` cell),
and `Δx` is the length of the cell centered on (Center, Face, Any) in `x` (a `v` cell).
`div_xyᶜᶜᵃ` ends up at the location `cca`.
"""
@inline function div_xyᶜᶜᵃ(i, j, k, grid, u, v)
    return 1 / Azᶜᶜᵃ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Δy_uᶠᶜᵃ, u) +
                                       δyᵃᶜᵃ(i, j, k, grid, Δx_vᶜᶠᵃ, v))
end
