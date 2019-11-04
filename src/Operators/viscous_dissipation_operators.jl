####
#### Viscous fluxes
####

@inline viscous_flux_ux(i, j, k, grid, u, ν_ccc) = ν_ccc * ℑxᶜᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * δxᶜᵃᵃ(i, j, k, grid, u)
@inline viscous_flux_uy(i, j, k, grid, u, ν_ffc) = ν_ffc * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * δyᵃᶠᵃ(i, j, k, grid, u)
@inline viscous_flux_uz(i, j, k, grid, u, ν_fcf) = ν_fcf * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ)  * δzᵃᵃᶠ(i, j, k, grid, u)

@inline viscous_flux_vx(i, j, k, grid, v, ν_ffc) = ν_ffc * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * δxᶠᵃᵃ(i, j, k, grid, v)
@inline viscous_flux_vy(i, j, k, grid, v, ν_ccc) = ν_ccc * ℑyᵃᶜᵃ(i, j, k, grid, Ayᵃᵃᶜ) * δyᵃᶜᵃ(i, j, k, grid, v)
@inline viscous_flux_vz(i, j, k, grid, v, ν_cff) = ν_cff * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ)  * δzᵃᵃᶠ(i, j, k, grid, v)

@inline viscous_flux_wx(i, j, k, grid, w, ν_fcf) = ν_fcf * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * δxᶠᵃᵃ(i, j, k, grid, w)
@inline viscous_flux_wy(i, j, k, grid, w, ν_cff) = ν_cff * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * δyᵃᶠᵃ(i, j, k, grid, w)
@inline viscous_flux_wz(i, j, k, grid, w, ν_ccc) = ν_ccc * ℑzᵃᵃᶜ(i, j, k, grid, Azᵃᵃᵃ)  * δzᵃᵃᶜ(i, j, k, grid, w)

####
#### Viscous dissipation operators
####

"""
    ∇ν∇u(i, j, k, grid::Grid, u::AbstractArray, ν::AbstractFloat)

Calculates viscous dissipation for the u-velocity via

    1/V * [δxᶠᵃᵃ(ν * ℑxᶜᵃᵃ(Ax) * δxᶜᵃᵃ(u)) + δyᵃᶜᵃ(ν * ℑyᵃᶠᵃ(Ay) * δyᵃᶠᵃ(u)) + δzᵃᵃᶜ(ν * ℑzᵃᵃᶠ(Az) * δzᵃᵃᶠ(u))]

which will end up at the location `fcc`.
"""
@inline function div_ν∇u(i, j, k, grid, ν, u)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, viscous_flux_ux, u, ν) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_uy, u, ν) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_uz, u, ν))
end

"""
    ∇ν∇v(i, j, k, grid::Grid, v::AbstractArray, ν::AbstractFloat)

Calculates viscous dissipation for the v-velocity via

    1/Vᵛ * [δxᶜᵃᵃ(ν * ℑxᶠᵃᵃ(Ax) * δxᶠᵃᵃ(v)) + δyᵃᶠᵃ(ν * ℑyᵃᶜᵃ(Ay) * δyᵃᶜᵃ(v)) + δzᵃᵃᶜ(ν * ℑzᵃᵃᶠ(Az) * δzᵃᵃᶠ(v))]

which will end up at the location `cfc`.
"""
@inline function div_ν∇v(i, j, k, grid, ν, v)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_vx, v, ν) +
                                    δyᵃᶠᵃ(i, j, k, grid, viscous_flux_vy, v, ν) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_vz, v, ν))
end

"""
    ∇ν∇w(i, j, k, grid::Grid, w::AbstractArray, ν::AbstractFloat)

Calculates viscous dissipation for the w-velocity via

    1/Vʷ * [δxᶜᵃᵃ(ν * ℑxᶠᵃᵃ(Ax) * δxᶠᵃᵃ(w)) + δyᵃᶜᵃ(ν * ℑyᵃᶠᵃ(Ay) * δyᵃᶠᵃ(w)) + δzᵃᵃᶠ(ν * ℑzᵃᵃᶜ(Az) * δzᵃᵃᶜ(w))]

which will end up at the location `ccf`.
"""
@inline function div_ν∇w(i, j, k, grid, ν, w)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_wx, w, ν) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_wy, w, ν) +
                                    δzᵃᵃᶠ(i, j, k, grid, viscous_flux_wz, w, ν))
end
