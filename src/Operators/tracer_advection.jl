# Calculate the flux of a tracer quantity c through the faces of a cell.
# In this case, the fluxes are given by u*Ax*T̅ˣ, v*Ay*T̅ʸ, and w*Az*T̅ᶻ.
@inline tracer_flux_x(i, j, k, grid, u, c) = @inbounds AxC(i, j, k, grid) * u[i, j, k] * ℑx_faa(i, j, k, grid, c)
@inline tracer_flux_y(i, j, k, grid, v, c) = @inbounds AyC(i, j, k, grid) * v[i, j, k] * ℑy_afa(i, j, k, grid, c)
@inline tracer_flux_z(i, j, k, grid, w, c) = @inbounds  Az(i, j, k, grid) * w[i, j, k] * ℑz_aaf(i, j, k, grid, c)

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

