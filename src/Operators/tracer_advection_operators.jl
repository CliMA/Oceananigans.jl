####
#### Advective tracer fluxes
####

# Calculate the flux of a tracer quantity c through the faces of a cell.
# In this case, the fluxes are given by u*Ax*T̅ˣ, v*Ay*T̅ʸ, and w*Az*T̅ᶻ.
@inline advective_tracer_flux_x(i, j, k, grid, u, c) = Ax_ψᵃᵃᶠ(i, j, k, grid, u) * ℑxᶠᵃᵃ(i, j, k, grid, c)
@inline advective_tracer_flux_y(i, j, k, grid, v, c) = Ay_ψᵃᵃᶠ(i, j, k, grid, v) * ℑyᵃᶠᵃ(i, j, k, grid, c)
@inline advective_tracer_flux_z(i, j, k, grid, w, c) = Az_ψᵃᵃᵃ(i, j, k, grid, w) * ℑzᵃᵃᶠ(i, j, k, grid, c)

####
#### Tracer advection operator
####

"""
    div_flux(i, j, k, grid, U, c)

Calculates the divergence of the flux of a tracer quantity c being advected by
a velocity field U = (u, v, w), ∇·(Uc),

    1/V * [δxᶜᵃᵃ(Ax * u * ℑxᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * v * ℑyᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * w * ℑzᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function div_uc(i, j, k, grid, U, c)
    1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, advective_tracer_flux_x, U.u, c) +
                             δyᵃᶜᵃ(i, j, k, grid, advective_tracer_flux_y, U.v, c) +
                             δzᵃᵃᶜ(i, j, k, grid, advective_tracer_flux_z, U.w, c))
end
