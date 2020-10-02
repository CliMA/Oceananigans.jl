#####
##### Tracer advection operator
#####

"""
    div_uc(i, j, k, grid, advection, U, c)

Calculates the divergence of the flux of a tracer quantity c being advected by
a velocity field U = (u, v, w), ∇·(Uc),

    1/V * [δxᶜᵃᵃ(Ax * u * ℑxᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * v * ℑyᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * w * ℑzᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function div_Uc(i, j, k, grid, advection, U, c)
    1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, advective_tracer_flux_x, advection, U.u, c) +
                             δyᵃᶜᵃ(i, j, k, grid, advective_tracer_flux_y, advection, U.v, c) +
                             δzᵃᵃᶜ(i, j, k, grid, advective_tracer_flux_z, advection, U.w, c))
end

#####
##### Non-conservative tracer advection operators for background fields
#####

"""
    U_grad_c(i, j, k, grid, advection, U, c)

Calculates the non-conservative advection of a tracer `c` by velocity field `U`, U·∇c,

    ℑxᶜᵃᵃ(U.u) * ℑxᶜᵃᵃ(δxᶠᵃᵃ(c)) + ℑyᵃᶜᵃ(U.v) * ℑyᵃᶜᵃ(δyᵃᶠᵃ(v)) + ℑzᵃᵃᶜ(U.w) * ℑzᵃᵃᶜ(δzᵃᵃᶠ(c)) 

which will end up at the location `ccc`.
"""
@inline function U_grad_c(i, j, k, grid, advection, U, c)
    return @inbounds (ℑxᶜᵃᵃ(i, j, k, grid, U.u) * ℑxᶜᵃᵃ(i, j, k, grid, δxᶠᵃᵃ, c) +
                      ℑyᵃᶜᵃ(i, j, k, grid, U.v) * ℑyᵃᶜᵃ(i, j, k, grid, δyᵃᶠᵃ, c) +
                      ℑzᵃᵃᶜ(i, j, k, grid, U.w) * ℑzᵃᵃᶜ(i, j, k, grid, δzᵃᵃᶠ, c))
end
