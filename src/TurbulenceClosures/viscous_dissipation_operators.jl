#####
#####Viscous fluxes
#####

@inline viscous_flux_ux(i, j, k, grid, ν_ccc, u) = ν_ccc * ℑxᶜᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶜᵃᵃ(i, j, k, grid, u)
@inline viscous_flux_uy(i, j, k, grid, ν_ffc, u) = ν_ffc * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶠᵃ(i, j, k, grid, u)
@inline viscous_flux_uz(i, j, k, grid, ν_fcf, u) = ν_fcf * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶠ(i, j, k, grid, u)

@inline viscous_flux_vx(i, j, k, grid, ν_ffc, v) = ν_ffc * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶠᵃᵃ(i, j, k, grid, v)
@inline viscous_flux_vy(i, j, k, grid, ν_ccc, v) = ν_ccc * ℑyᵃᶜᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶜᵃ(i, j, k, grid, v)
@inline viscous_flux_vz(i, j, k, grid, ν_cff, v) = ν_cff * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶠ(i, j, k, grid, v)

@inline viscous_flux_wx(i, j, k, grid, ν_fcf, w) = ν_fcf * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶠᵃᵃ(i, j, k, grid, w)
@inline viscous_flux_wy(i, j, k, grid, ν_cff, w) = ν_cff * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶠᵃ(i, j, k, grid, w)
@inline viscous_flux_wz(i, j, k, grid, ν_ccc, w) = ν_ccc * ℑzᵃᵃᶜ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶜ(i, j, k, grid, w)

#####
#####Viscous dissipation operators
#####

"""
    ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, νˣ, νʸ, νᶻ, u)

Calculates viscous dissipation for the u-velocity via

    1/V * [δxᶠᵃᵃ(νˣ * ℑxᶜᵃᵃ(Ax) * δxᶜᵃᵃ(u)) + δyᵃᶜᵃ(νʸ * ℑyᵃᶠᵃ(Ay) * δyᵃᶠᵃ(u)) + δzᵃᵃᶜ(νᶻ * ℑzᵃᵃᶠ(Az) * δzᵃᵃᶠ(u))]

which will end up at the location `fcc`.
"""
@inline function ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, νˣ, νʸ, νᶻ, u)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, viscous_flux_ux, νˣ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_uy, νʸ, u) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_uz, νᶻ, u))
end

"""
    ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, νˣ, νʸ, νᶻ, v)

Calculates viscous dissipation for the v-velocity via

    1/Vᵛ * [δxᶜᵃᵃ(νˣ * ℑxᶠᵃᵃ(Ax) * δxᶠᵃᵃ(v)) + δyᵃᶠᵃ(νʸ * ℑyᵃᶜᵃ(Ay) * δyᵃᶜᵃ(v)) + δzᵃᵃᶜ(νᶻ * ℑzᵃᵃᶠ(Az) * δzᵃᵃᶠ(v))]

which will end up at the location `cfc`.
"""
@inline function ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, νˣ, νʸ, νᶻ, v)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_vx, νˣ, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, viscous_flux_vy, νʸ, v) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_vz, νᶻ, v))
end

"""
    ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid::Grid, νˣ, νʸ, νᶻ, w)

Calculates viscous dissipation for the w-velocity via

    1/Vʷ * [δxᶜᵃᵃ(νˣ * ℑxᶠᵃᵃ(Ax) * δxᶠᵃᵃ(w)) + δyᵃᶜᵃ(νʸ * ℑyᵃᶠᵃ(Ay) * δyᵃᶠᵃ(w)) + δzᵃᵃᶠ(νᶻ * ℑzᵃᵃᶜ(Az) * δzᵃᵃᶜ(w))]

which will end up at the location `ccf`.
"""
@inline function ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, νˣ, νʸ, νᶻ, w)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_wx, νˣ, w) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_wy, νʸ, w) +
                                    δzᵃᵃᶠ(i, j, k, grid, viscous_flux_wz, νᶻ, w))
end

#####
#####Viscous dissipation for isotropic viscosity
#####

@inline ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, ν, u) = ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, ν, ν, ν, u)
@inline ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, ν, v) = ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, ν, ν, ν, v)
@inline ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, ν, w) = ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, ν, ν, ν, w)
