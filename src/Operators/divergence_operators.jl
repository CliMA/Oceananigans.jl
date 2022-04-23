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
    return 1 / Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, u) +
                                      δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, v) +
                                      δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, w))
end

"""
    div_xyᶜᶜᵃ(i, j, k, grid, u, v)

Returns the discrete `div_xy = ∂x u + ∂y v` of velocity field `u, v` defined as

```
1 / Azᶜᶜᵃ * [δxᶜᵃᵃ(Δyᵃᶜᵃ * u) + δyᵃᶜᵃ(Δxᶜᵃᵃ * v)]
```

at `i, j, k`, where `Azᶜᶜᵃ` is the area of the cell centered on (Center, Center, Any) --- a tracer cell,
`Δy` is the length of the cell centered on (Face, Center, Any) in `y` (a `u` cell),
and `Δx` is the length of the cell centered on (Center, Face, Any) in `x` (a `v` cell).
`div_xyᶜᶜᵃ` ends up at the location `cca`.
"""
@inline function div_xyᶜᶜᶜ(i, j, k, grid, u, v)
    return 1 / Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u) +
                                       δyᵃᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v))
end

# Convention
 index_left(i, ::Center) = i
 index_left(i, ::Face)   = i - 1
index_right(i, ::Center) = i + 1
index_right(i, ::Face)   = i

@inline Base.div(i, j, k, grid::AbstractGrid, loc, q_west, q_east, q_south, q_north, q_bottom, q_top) =
    1 / volume(i, j, k, grid, loc...) * (δx_Ax_q(i, j, k, grid, loc, q_west, q_east) + 
                                         δy_Ay_q(i, j, k, grid, loc, q_south, q_north) + 
                                         δz_Az_q(i, j, k, grid, loc, q_bottom, q_top))
    
@inline function δx_Ax_q(i, j, k, grid, (LX, LY, LZ), qᵂ, qᴱ)
    iᵂ = index_left(i, LX)
    Axᵂ = Ax(iᵂ, j, k, grid, LX, LY, LZ)

    iᴱ = index_right(i, LX)
    Axᴱ = Ax(iᴱ, j, k, grid, LX, LY, LZ)

    return Axᴱ * qᴱ - Axᵂ * qᵂ
end

@inline function δy_Ay_q(i, j, k, grid, (LX, LY, LZ), qˢ, qᴺ)
    jˢ = index_left(j, LY)
    Ayˢ = Ay(i, jˢ, k, grid, LX, LY, LZ)

    jᴺ = index_right(j, LY)
    Ayᴺ = Ay(i, jᴺ, k, grid, LX, LY, LZ)

    return Ayᴺ * qᴺ - Ayˢ * qˢ
end

@inline function δz_Az_q(i, j, k, grid, (LX, LY, LZ), qᴮ, qᵀ)
    kᴮ = index_left(k, LZ)
    Azᴮ = Az(i, j, kᴮ, grid, LX, LY, LZ)

    kᵀ = index_right(k, LZ)
    Azᵀ = Az(i, j, kᵀ, grid, LX, LY, LZ)

    return Azᵀ * qᵀ - Azᴮ * qᴮ
end

# And flat!

@inline δx_Ax_q(i, j, k, grid::XFlatGrid, args...) = zero(grid)
@inline δy_Ay_q(i, j, k, grid::YFlatGrid, args...) = zero(grid)
@inline δz_Az_q(i, j, k, grid::ZFlatGrid, args...) = zero(grid)
