# Calculate the flux of a tracer quantity c through the faces of a cell.
# In this case, the fluxes are given by u*Ax*T̅ˣ, v*Ay*T̅ʸ, and w*Az*T̅ᶻ.
@inline tracer_flux_x(i, j, k, grid, u, c) = Ax_u(i, j, k, grid, u) * ℑxᶠᵃᵃ(i, j, k, grid, c)
@inline tracer_flux_y(i, j, k, grid, v, c) = Ay_v(i, j, k, grid, v) * ℑyᵃᶠᵃ(i, j, k, grid, c)
@inline tracer_flux_z(i, j, k, grid, w, c) = Az_w(i, j, k, grid, w) * ℑzᵃᵃᶠ(i, j, k, grid, c)

"""
    div_flux(i, j, k, grid, U, c)

Calculates the divergence of the flux of a tracer quantity c being advected by
a velocity field U = (u, v, w), ∇·(Uc),

    1/V * [δxᶜᵃᵃ(Ax * u * ℑxᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * v * ℑyᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * w * ℑzᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function div_flux(i, j, k, grid, u, v, w, c)
    1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, tracer_flux_x, u, c) +
                             δyᵃᶜᵃ(i, j, k, grid, tracer_flux_y, v, c) +
                             δzᵃᵃᶜ(i, j, k, grid, tracer_flux_z, w, c))
end

