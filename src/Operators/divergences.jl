####
#### Divergence operators
####

"""
    div_cc(i, j, k, grid, u, v)

Calculates the horizontal divergence ∇ₕ·(u, v) of a 2D velocity field (u, v) via

    1/V * [δx_caa(Ax * u) + δy_aca(Ay * v)]

which will end up at the location `cca`.
"""
@inline function div_cc(i, j, k, grid, u, v)
    1/VC(i, j, k, grid) * (δx_caa(i, j, k, grid, Ax_u, u) +
                           δy_aca(i, j, k, grid, Ay_v, v))
end

"""
    div_ccc(i, j, k, grid, u, v, w)

Calculates the divergence ∇·U of a vector field U = (u, v, w),

    1/V * [δx_caa(Ax * u) + δx_aca(Ay * v) + δz_aac(Az * w)],

which will end up at the cell centers `ccc`.
"""
@inline function div_ccc(i, j, k, grid, u, v, w)
    1/VC(i, j, k, grid) * (δx_caa(i, j, k, grid, Ax_u, u) +
                           δy_aca(i, j, k, grid, Ay_v, v) +
                           δz_aac(i, j, k, grid, Az_w, w))
end

