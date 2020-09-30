#####
##### Momentum advection operators
#####

"""
    div_ũu(i, j, k, grid, advection, U)

Calculate the advection of momentum in the x-direction U·∇u

    1/Vᵘ * [δxᶠᵃᵃ(ℑxᶜᵃᵃ(Ax * u) * ℑxᶜᵃᵃ(u)) + δy_fca(ℑxᶠᵃᵃ(Ay * v) * ℑyᵃᶠᵃ(u)) + δz_fac(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(u))]

which will end up at the location `fcc`.
"""
@inline function div_ũu(i, j, k, grid, advection, U, u)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_uu, advection, U.u, u)    +
                                    δyᵃᶜᵃ(i, j, k, grid, momentum_flux_uv, advection, U.v, u) +
                                    δzᵃᵃᶜ(i, j, k, grid, momentum_flux_uw, advection, U.w, u))
end

"""
    div_ũv(i, j, k, grid, advection, U)

Calculates the advection of momentum in the y-direction U·∇v

    1/Vʸ * [δx_cfa(ℑyᵃᶠᵃ(Ax * u) * ℑxᶠᵃᵃ(v)) + δyᵃᶠᵃ(ℑyᵃᶜᵃ(Ay * v) * ℑyᵃᶜᵃ(v)) + δz_afc(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(w))]

which will end up at the location `cfc`.
"""
@inline function div_ũv(i, j, k, grid, advection, U, v)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_vu, advection, U.u, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, momentum_flux_vv, advection, U.v, v)    +
                                    δzᵃᵃᶜ(i, j, k, grid, momentum_flux_vw, advection, U.w, v))
end

"""
    div_ũw(i, j, k, grid, advection, U)

Calculates the advection of momentum in the z-direction U·∇w

    1/Vʷ * [δx_caf(ℑzᵃᵃᶠ(Ax * u) * ℑxᶠᵃᵃ(w)) + δy_acf(ℑzᵃᵃᶠ(Ay * v) * ℑyᵃᶠᵃ(w)) + δzᵃᵃᶠ(ℑzᵃᵃᶜ(Az * w) * ℑzᵃᵃᶜ(w))]

which will end up at the location `ccf`.
"""
@inline function div_ũw(i, j, k, grid, advection, U, w)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_wu, advection, U.u, w) +
                                    δyᵃᶜᵃ(i, j, k, grid, momentum_flux_wv, advection, U.v, w) +
                                    δzᵃᵃᶠ(i, j, k, grid, momentum_flux_ww, advection, U.w, w))
end
