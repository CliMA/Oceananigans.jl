####
#### Calculating fluxes
####

# Calculate the flux of a tracer quantity c through the faces of a cell.
# In this case, the fluxes are given by u*Ax*T̅ˣ, v*Ay*T̅ʸ, and w*Az*T̅ᶻ.
@inline tracer_flux_x(i, j, k, grid, u, c) = @inbounds AxC(i, j, k, grid) * u[i, j, k] * ℑx_faa(i, j, k, grid, c)
@inline tracer_flux_y(i, j, k, grid, v, c) = @inbounds AyC(i, j, k, grid) * v[i, j, k] * ℑy_afa(i, j, k, grid, c)
@inline tracer_flux_z(i, j, k, grid, w, c) = @inbounds  Az(i, j, k, grid) * w[i, j, k] * ℑz_aaf(i, j, k, grid, c)

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
    1/V(i, j, k, grid) * (δx_caa(i, j, k, grid, Ax_u, u) +
                          δy_aca(i, j, k, grid, fy_v, v))
end

"""
    div_ccc(i, j, k, grid, fx, fy, fz)

Calculates the divergence ∇·U of a vector field U = (u, v, w),

    1/V * [δx_caa(Ax * fx) + δx_aca(Ay * fy) + δz_aac(Az * fz)],

which will end up at the cell centers `ccc`.
"""
@inline function div_ccc(i, j, k, grid, u, v, w)
    1/V(i, j, k, grid) * (δx_caa(i, j, k, grid, Ax_u, fx) +
                          δy_aca(i, j, k, grid, Ay_v, fy) +
                          δz_aac(i, j, k, grid, Az_w, fz))
end

"""
    div_flux(i, j, k, grid, U, c)

Calculates the divergence of the flux of a tracer quantity c being advected by
a velocity field U = (u, v, w), ∇·(Uc),

    1/V * [δx_caa(Ax * u * ℑx_faa(c)) + δy_aca(Ay * v * ℑy_afa(c)) + δz_aac(Az * w * ℑz_aaf(c))]

which will end up at the location `ccc`.
"""
@inline function div_flux(i, j, k, grid, u, v, w, c)
    1/V(i, j, k, grid) * (δx_caa(i, j, k, grid, tracer_flux_x, u, c) +
                          δy_aca(i, j, k, grid, tracer_flux_y, v, c) +
                          δz_aac(i, j, k, grid, tracer_flux_z, w, c))
end

