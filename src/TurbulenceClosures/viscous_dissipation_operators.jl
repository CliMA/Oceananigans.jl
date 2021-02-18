#####
##### Viscosities at different locations
#####

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::Number) = ν

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::Function) = ν(xnode(Center, grid, i), ynode(Center, grid, j), znode(Center, grid, k), clock.time)
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::Function) = ν(xnode(Face,   grid, i), ynode(Face,   grid, j), znode(Center, grid, k), clock.time)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::Function) = ν(xnode(Face,   grid, i), ynode(Center, grid, j), znode(Face,   grid, k), clock.time)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::Function) = ν(xnode(Face,   grid, i), ynode(Center, grid, j), znode(Face,   grid, k), clock.time)

#####
##### Viscous fluxes
#####

@inline viscous_flux_ux(i, j, k, grid, clock, ν, u) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * ℑxᶜᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶜᵃᵃ(i, j, k, grid, u)
@inline viscous_flux_uy(i, j, k, grid, clock, ν, u) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶠᵃ(i, j, k, grid, u)
@inline viscous_flux_uz(i, j, k, grid, clock, ν, u) = νᶠᶜᶠ(i, j, k, grid, clock, ν) * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶠ(i, j, k, grid, u)

@inline viscous_flux_vx(i, j, k, grid, clock, ν, v) = νᶠᶠᶜ(i, j, k, grid, clock, ν) * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶠᵃᵃ(i, j, k, grid, v)
@inline viscous_flux_vy(i, j, k, grid, clock, ν, v) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * ℑyᵃᶜᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶜᵃ(i, j, k, grid, v)
@inline viscous_flux_vz(i, j, k, grid, clock, ν, v) = νᶜᶠᶠ(i, j, k, grid, clock, ν) * ℑzᵃᵃᶠ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶠ(i, j, k, grid, v)

@inline viscous_flux_wx(i, j, k, grid, clock, ν, w) = νᶠᶜᶠ(i, j, k, grid, clock, ν) * ℑxᶠᵃᵃ(i, j, k, grid, Axᵃᵃᶜ) * ∂xᶠᵃᵃ(i, j, k, grid, w)
@inline viscous_flux_wy(i, j, k, grid, clock, ν, w) = νᶜᶠᶠ(i, j, k, grid, clock, ν) * ℑyᵃᶠᵃ(i, j, k, grid, Ayᵃᵃᶜ) * ∂yᵃᶠᵃ(i, j, k, grid, w)
@inline viscous_flux_wz(i, j, k, grid, clock, ν, w) = νᶜᶜᶜ(i, j, k, grid, clock, ν) * ℑzᵃᵃᶜ(i, j, k, grid, Azᵃᵃᵃ) * ∂zᵃᵃᶜ(i, j, k, grid, w)

#####
##### Viscous dissipation operators
#####

"""
    ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, νˣ, νʸ, νᶻ, u)

Calculates viscous dissipation for the u-velocity via

    1/V * [δxᶠᵃᵃ(νˣ * ℑxᶜᵃᵃ(Ax) * δxᶜᵃᵃ(u)) + δyᵃᶜᵃ(νʸ * ℑyᵃᶠᵃ(Ay) * δyᵃᶠᵃ(u)) + δzᵃᵃᶜ(νᶻ * ℑzᵃᵃᶠ(Az) * δzᵃᵃᶠ(u))]

which will end up at the location `fcc`.
"""
@inline function ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, clock, νˣ, νʸ, νᶻ, u)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, viscous_flux_ux, clock, νˣ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_uy, clock, νʸ, u) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_uz, clock, νᶻ, u))
end

"""
    ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, νˣ, νʸ, νᶻ, v)

Calculates viscous dissipation for the v-velocity via

    1/Vᵛ * [δxᶜᵃᵃ(νˣ * ℑxᶠᵃᵃ(Ax) * δxᶠᵃᵃ(v)) + δyᵃᶠᵃ(νʸ * ℑyᵃᶜᵃ(Ay) * δyᵃᶜᵃ(v)) + δzᵃᵃᶜ(νᶻ * ℑzᵃᵃᶠ(Az) * δzᵃᵃᶠ(v))]

which will end up at the location `cfc`.
"""
@inline function ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, clock, νˣ, νʸ, νᶻ, v)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_vx, clock, νˣ, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, viscous_flux_vy, clock, νʸ, v) +
                                    δzᵃᵃᶜ(i, j, k, grid, viscous_flux_vz, clock, νᶻ, v))
end

"""
    ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid::Grid, νˣ, νʸ, νᶻ, w)

Calculates viscous dissipation for the w-velocity via

    1/Vʷ * [δxᶜᵃᵃ(νˣ * ℑxᶠᵃᵃ(Ax) * δxᶠᵃᵃ(w)) + δyᵃᶜᵃ(νʸ * ℑyᵃᶠᵃ(Ay) * δyᵃᶠᵃ(w)) + δzᵃᵃᶠ(νᶻ * ℑzᵃᵃᶜ(Az) * δzᵃᵃᶜ(w))]

which will end up at the location `ccf`.
"""
@inline function ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, clock, νˣ, νʸ, νᶻ, w)
    return 1/Vᵃᵃᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, viscous_flux_wx, clock, νˣ, w) +
                                    δyᵃᶜᵃ(i, j, k, grid, viscous_flux_wy, clock, νʸ, w) +
                                    δzᵃᵃᶠ(i, j, k, grid, viscous_flux_wz, clock, νᶻ, w))
end

#####
##### Viscous dissipation for isotropic viscosity
#####

@inline ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, clock, ν, u) = ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, clock, ν, ν, ν, u)
@inline ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, clock, ν, v) = ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, clock, ν, ν, ν, v)
@inline ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, clock, ν, w) = ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, clock, ν, ν, ν, w)
