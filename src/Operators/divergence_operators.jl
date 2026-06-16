#####
##### Divergence operators
#####

"""
$(TYPEDSIGNATURES)

Calculate the divergence ``𝛁·𝐕`` of a vector field ``𝐕 = (u, v, w)``,

```julia
1/V * [δxᶜᵃᵃ(Ax * u) + δxᵃᶜᵃ(Ay * v) + δzᵃᵃᶜ(Az * w)]
```

which ends up at the cell centers `ccc`.
"""
@inline divᶜᶜᶜ(i, j, k, grid, u, v, w) =
    V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, u) +
                             δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, v) +
                             δzᶜᶜᶜ(i, j, k, grid, Az_qᶜᶜᶠ, w))

"""
$(TYPEDSIGNATURES)

Return the discrete horizontal flux divergence `δxᶜᶜᶜ(Ax * u) + δyᶜᶜᶜ(Ay * v)` of the
velocity field `u, v`, where `Ax` and `Ay` are the areas of the cell faces in `x` and `y`.
Unlike `div_xyᶜᶜᶜ`, this is _not_ divided by the cell volume.

`flux_div_xyᶜᶜᶜ` ends up at the location `ccc`.
"""
@inline flux_div_xyᶜᶜᶜ(i, j, k, grid, u, v) = (δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, u) +
                                               δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, v))

@inline div_xyᶜᶜᶜ(i, j, k, grid, u, v) =
    V⁻¹ᶜᶜᶜ(i, j, k, grid) * flux_div_xyᶜᶜᶜ(i, j, k, grid, u, v)

@inline div_xyᶜᶜᶠ(i, j, k, grid, Qu, Qv) =
    V⁻¹ᶜᶜᶠ(i, j, k, grid) * (δxᶜᶜᶠ(i, j, k, grid, Ay_qᶠᶜᶠ, Qu) +
                             δyᶜᶜᶠ(i, j, k, grid, Ax_qᶜᶠᶠ, Qv))

# Convention
index_left(i, ::Center)  = i
index_left(i, ::Face)    = i - 1
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
