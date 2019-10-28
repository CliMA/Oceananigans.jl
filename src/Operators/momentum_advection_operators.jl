####
#### Momentum fluxes
####

@inline momentum_flux_uu(i, j, k, grid, u)    = ℑx_caa(i, j, k, grid, Ax_u, u) * ℑx_caa(i, j, k, grid, u)
@inline momentum_flux_uv(i, j, k, grid, u, v) = ℑx_faa(i, j, k, grid, Ay_v, v) * ℑy_afa(i, j, k, grid, u)
@inline momentum_flux_uw(i, j, k, grid, u, w) = ℑx_faa(i, j, k, grid, Az_w, w) * ℑz_aaf(i, j, k, grid, u)

@inline momentum_flux_vu(i, j, k, grid, u, v) = ℑy_afa(i, j, k, grid, Ax_u, u) * ℑx_faa(i, j, k, grid, v)
@inline momentum_flux_vv(i, j, k, grid, v)    = ℑy_aca(i, j, k, grid, Ay_v, v) * ℑy_aca(i, j, k, grid, v)
@inline momentum_flux_vw(i, j, k, grid, v, w) = ℑy_afa(i, j, k, grid, Az_w, w) * ℑz_aaf(i, j, k, grid, v)

@inline momentum_flux_wu(i, j, k, grid, u, w) = ℑz_aaf(i, j, k, grid, Ax_u, u) * ℑx_faa(i, j, k, grid, w)
@inline momentum_flux_wv(i, j, k, grid, v, w) = ℑz_aaf(i, j, k, grid, Ay_v, v) * ℑy_afa(i, j, k, grid, w)
@inline momentum_flux_ww(i, j, k, grid, w)    = ℑz_aac(i, j, k, grid, Az_w, w) * ℑz_aac(i, j, k, grid, w)

####
#### Momentum advection operators
####

"""
    u∇u(i, j, k, grid, u, v, w)

Calculate the advection of momentum in the x-direction U·∇u

    1/Vᵘ * [δx_faa(ℑx_caa(Ax * u) * ℑx_caa(u)) + δy_fca(ℑx_faa(Ay * v) * ℑy_afa(u)) + δz_fac(ℑx_faa(Az * w) * ℑz_aaf(u))]

which will end up at the location `fcc`.
"""
@inline function u∇u(i, j, k, grid, u, v, w)
    1/Vᵘ(i, j, k, grid) * (δx_faa(i, j, k, grid, momentum_flux_uu, u)    +
                           δy_aca(i, j, k, grid, momentum_flux_uv, u, v) +
                           δz_aac(i, j, k, grid, momentum_flux_uw, u, w))
end

"""
    u∇v(i, j, k, grid, u, v, w)

Calculates the advection of momentum in the y-direction U·∇v

    1/Vʸ * [δx_cfa(ℑy_afa(Ax * u) * ℑx_faa(v)) + δy_afa(ℑy_aca(Ay * v) * ℑy_aca(v)) + δz_afc(ℑx_faa(Az * w) * ℑz_aaf(w))]

which will end up at the location `cfc`.
"""
@inline function u∇v(i, j, k, grid, u, v, w)
    1/Vᵛ(i, j, k, grid) * (δx_caa(i, j, k, grid, momentum_flux_vu, u, v) +
                           δy_afa(i, j, k, grid, momentum_flux_vv, v)    +
                           δz_aac(i, j, k, grid, momentum_flux_vw, v, w))
end

"""
    u∇w(i, j, k, grid, u, v, w)

Calculates the advection of momentum in the z-direction U·∇w

    1/Vʷ * [δx_caf(ℑz_aaf(Ax * u) * ℑx_faa(w)) + δy_acf(ℑz_aaf(Ay * v) * ℑy_afa(w)) + δz_aaf(ℑz_aac(Az * w) * ℑz_aac(w))]

which will end up at the location `ccf`.
"""
@inline function u∇w(i, j, k, grid, u, v, w)
    1/Vʷ(i, j, k, grid) * (δx_caa(i, j, k, grid, momentum_flux_wu, u, w) +
                           δy_aca(i, j, k, grid, momentum_flux_wv, v, w) +
                           δz_aaf(i, j, k, grid, momentum_flux_ww, w))
end
