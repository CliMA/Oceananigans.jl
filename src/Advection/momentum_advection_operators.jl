#####
##### Momentum advection operators
#####

"""
    div_ũu(i, j, k, grid, advection, U, u)

Calculate the advection of momentum in the x-direction using the conserative form, ∇·(Uu)

    1/Vᵘ * [δxᶠᵃᵃ(ℑxᶜᵃᵃ(Ax * u) * ℑxᶜᵃᵃ(u)) + δy_fca(ℑxᶠᵃᵃ(Ay * v) * ℑyᵃᶠᵃ(u)) + δz_fac(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(u))]

which will end up at the location `fcc`.
"""
@inline function div_Uu(i, j, k, grid, advection, U, u)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_uu, advection, U.u, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, momentum_flux_uv, advection, U.v, u) +
                                    δzᵃᵃᶜ(i, j, k, grid, momentum_flux_uw, advection, U.w, u))
end

"""
    div_ũv(i, j, k, grid, advection, U, v)

Calculate the advection of momentum in the y-direction using the conserative form, ∇·(Uv)

    1/Vʸ * [δx_cfa(ℑyᵃᶠᵃ(Ax * u) * ℑxᶠᵃᵃ(v)) + δyᵃᶠᵃ(ℑyᵃᶜᵃ(Ay * v) * ℑyᵃᶜᵃ(v)) + δz_afc(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(w))]

which will end up at the location `cfc`.
"""
@inline function div_Uv(i, j, k, grid, advection, U, v)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_vu, advection, U.u, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, momentum_flux_vv, advection, U.v, v)    +
                                    δzᵃᵃᶜ(i, j, k, grid, momentum_flux_vw, advection, U.w, v))
end

"""
    div_ũw(i, j, k, grid, advection, U, w)

Calculate the advection of momentum in the z-direction using the conserative form, ∇·(Uw)

    1/Vʷ * [δx_caf(ℑzᵃᵃᶠ(Ax * u) * ℑxᶠᵃᵃ(w)) + δy_acf(ℑzᵃᵃᶠ(Ay * v) * ℑyᵃᶠᵃ(w)) + δzᵃᵃᶠ(ℑzᵃᵃᶜ(Az * w) * ℑzᵃᵃᶜ(w))]

which will end up at the location `ccf`.
"""
@inline function div_Uw(i, j, k, grid, advection, U, w)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_wu, advection, U.u, w) +
                                    δyᵃᶜᵃ(i, j, k, grid, momentum_flux_wv, advection, U.v, w) +
                                    δzᵃᵃᶠ(i, j, k, grid, momentum_flux_ww, advection, U.w, w))
end

#####
##### Non-conservative momentum advection operators for background fields
#####

"""
    U_grad_u(i, j, k, grid, U, u)

Calculate the advection of momentum in the x-direction with the "non-conservative" form, U·∇u,

    U.u * ℑxᶠᵃᵃ(δxᶜᵃᵃ(u)) + ℑxyᶠᶜᵃ(U.v) * ℑyᵃᶜᵃ(δyᵃᶠᵃ(u)) + ℑxzᶠᵃᶜ(U.w) * ℑzᵃᵃᶜ(δzᵃᵃᶠ(u)) 

which will end up at the location `fcc`.
"""
@inline function U_grad_u(i, j, k, grid, advection, U, u)
    return @inbounds (                      U.u[i, j, k] * ℑxᶠᵃᵃ(i, j, k, grid, δxᶜᵃᵃ, u) +
                      ℑxyᶠᶜᵃ(i, j, k, grid, U.v)         * ℑyᵃᶜᵃ(i, j, k, grid, δyᵃᶠᵃ, u) +
                      ℑxzᶠᵃᶜ(i, j, k, grid, U.w)         * ℑzᵃᵃᶜ(i, j, k, grid, δzᵃᵃᶠ, u))
end

"""
    U_grad_v(i, j, k, grid, U, u)

Calculate the advection of momentum in the y-direction with the "non-conservative" form, U·∇v,

    ℑxyᶜᶠᵃ(U.u) * ℑxᶜᵃᵃ(δxᶠᵃᵃ(v)) + U.v * ℑyᵃᶠᵃ(δyᵃᶜᵃ(v)) + ℑyzᵃᶠᶜ(U.w) * ℑzᵃᵃᶜ(δzᵃᵃᶠ(v)) 

which will end up at the location `fcc`.
"""
@inline function U_grad_v(i, j, k, grid, advection, U, v)
    return @inbounds (ℑxyᶜᶠᵃ(i, j, k, grid, U.u)         * ℑxᶜᵃᵃ(i, j, k, grid, δxᶠᵃᵃ, v) +
                                            U.v[i, j, k] * ℑyᵃᶠᵃ(i, j, k, grid, δyᵃᶜᵃ, v) +
                      ℑyzᵃᶠᶜ(i, j, k, grid, U.w)         * ℑzᵃᵃᶜ(i, j, k, grid, δzᵃᵃᶠ, v))
end

"""
    U_grad_w(i, j, k, grid, U, w)

Calculate the advection of momentum in the y-direction with the "non-conservative" form, U·∇w,

    ℑxzᶜᵃᶠ(U.u) * ℑxᶜᵃᵃ(δxᶠᵃᵃ(w)) + ℑyzᵃᶜᶠ(U.v) * ℑyᵃᶜᵃ(δyᵃᶠᵃ(w)) + U.w * ℑzᵃᵃᶠ(δzᵃᵃᶜ(w)) 

which will end up at the location `fcc`.
"""
@inline function U_grad_w(i, j, k, grid, advection, U, w)
    return @inbounds (ℑxzᶜᵃᶠ(i, j, k, grid, U.u)         * ℑxᶜᵃᵃ(i, j, k, grid, δxᶠᵃᵃ, w) +
                      ℑyzᵃᶜᶠ(i, j, k, grid, U.v)         * ℑyᵃᶜᵃ(i, j, k, grid, δyᵃᶠᵃ, w) +
                                            U.w[i, j, k] * ℑzᵃᵃᶠ(i, j, k, grid, δzᵃᵃᶜ, w))
end
