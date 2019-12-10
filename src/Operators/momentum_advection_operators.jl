####
#### Momentum fluxes
####

@inline momentum_flux_uu(i, j, k, grid, u)    = ℑxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline momentum_flux_uv(i, j, k, grid, u, v) = ℑxᶠᵃᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * ℑyᵃᶠᵃ(i, j, k, grid, u)
@inline momentum_flux_uw(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * ℑzᵃᵃᶠ(i, j, k, grid, u)

@inline momentum_flux_vu(i, j, k, grid, u, v) = ℑyᵃᶠᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * ℑxᶠᵃᵃ(i, j, k, grid, v)
@inline momentum_flux_vv(i, j, k, grid, v)    = ℑyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline momentum_flux_vw(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * ℑzᵃᵃᶠ(i, j, k, grid, v)

@inline momentum_flux_wu(i, j, k, grid, u, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * ℑxᶠᵃᵃ(i, j, k, grid, w)
@inline momentum_flux_wv(i, j, k, grid, v, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * ℑyᵃᶠᵃ(i, j, k, grid, w)
@inline momentum_flux_ww(i, j, k, grid, w)    = ℑzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * ℑzᵃᵃᶜ(i, j, k, grid, w)

####
#### Momentum advection operators
####

"""
    u∇u(i, j, k, grid, U)

Calculate the advection of momentum in the x-direction U·∇u

    1/Vᵘ * [δxᶠᵃᵃ(ℑxᶜᵃᵃ(Ax * u) * ℑxᶜᵃᵃ(u)) + δy_fca(ℑxᶠᵃᵃ(Ay * v) * ℑyᵃᶠᵃ(u)) + δz_fac(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(u))]

which will end up at the location `fcc`.
"""
@inline function div_ũu(i, j, k, grid, U)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_uu, U.u)    +
                                    δyᵃᶜᵃ(i, j, k, grid, momentum_flux_uv, U.u, U.v) +
                                    δzᵃᵃᶜ(i, j, k, grid, momentum_flux_uw, U.u, U.w))
end

"""
    u∇v(i, j, k, grid, U)

Calculates the advection of momentum in the y-direction U·∇v

    1/Vʸ * [δx_cfa(ℑyᵃᶠᵃ(Ax * u) * ℑxᶠᵃᵃ(v)) + δyᵃᶠᵃ(ℑyᵃᶜᵃ(Ay * v) * ℑyᵃᶜᵃ(v)) + δz_afc(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(w))]

which will end up at the location `cfc`.
"""
@inline function div_ũv(i, j, k, grid, U)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_vu, U.u, U.v) +
                                    δyᵃᶠᵃ(i, j, k, grid, momentum_flux_vv, U.v)    +
                                    δzᵃᵃᶜ(i, j, k, grid, momentum_flux_vw, U.v, U.w))
end

"""
    u∇w(i, j, k, grid, U)

Calculates the advection of momentum in the z-direction U·∇w

    1/Vʷ * [δx_caf(ℑzᵃᵃᶠ(Ax * u) * ℑxᶠᵃᵃ(w)) + δy_acf(ℑzᵃᵃᶠ(Ay * v) * ℑyᵃᶠᵃ(w)) + δzᵃᵃᶠ(ℑzᵃᵃᶜ(Az * w) * ℑzᵃᵃᶜ(w))]

which will end up at the location `ccf`.
"""
@inline function div_ũw(i, j, k, grid, U)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_wu, U.u, U.w) +
                                    δyᵃᶜᵃ(i, j, k, grid, momentum_flux_wv, U.v, U.w) +
                                    δzᵃᵃᶠ(i, j, k, grid, momentum_flux_ww, U.w))
end
