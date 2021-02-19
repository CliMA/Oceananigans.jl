#####
##### Viscosities at different locations
#####

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::Number) = ν
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::Number) = ν

@inline νᶜᶜᶜ(i, j, k, grid, clock, ν::Function) = ν(xnode(Center, i, grid), ynode(Center, j, grid), znode(Center, k, grid), clock.time)
@inline νᶠᶠᶜ(i, j, k, grid, clock, ν::Function) = ν(xnode(Face,   i, grid), ynode(Face,   j, grid), znode(Center, k, grid), clock.time)
@inline νᶠᶜᶠ(i, j, k, grid, clock, ν::Function) = ν(xnode(Face,   i, grid), ynode(Center, j, grid), znode(Face,   k, grid), clock.time)
@inline νᶜᶠᶠ(i, j, k, grid, clock, ν::Function) = ν(xnode(Face,   i, grid), ynode(Center, j, grid), znode(Face,   k, grid), clock.time)

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

#####
##### Products of viscosity and divergence, vorticity, and vertical momentum gradients
#####

@inline ν_δᶜᶜᶜ(i, j, k, grid, clock, ν, u, v) = @inbounds νᶜᶜᶜ(i, j, k, grid, clock, ν) * div_xyᶜᶜᵃ(i, j, k, grid, u, v)
@inline ν_ζᶠᶠᶜ(i, j, k, grid, clock, ν, u, v) = @inbounds νᶠᶠᶜ(i, j, k, grid, clock, ν) * ζ₃ᶠᶠᵃ(i, j, k, grid, u, v)

@inline ν_uzᶠᶜᶠ(i, j, k, grid, clock, ν, u) = @inbounds νᶠᶜᶠ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶠ(i, j, k, grid, u)
@inline ν_vzᶜᶠᶠ(i, j, k, grid, clock, ν, v) = @inbounds νᶜᶠᶠ(i, j, k, grid, clock, ν) * ∂zᵃᵃᶠ(i, j, k, grid, v)

#####
##### Viscosity-stress products at various locations
#####

"""
    ν_σᶜᶜᶜ(i, j, k, grid, ν, σᶜᶜᶜ, u, v, w)

Multiply the array `ν` located at `ᶜᶜᶜ` by a function

    `σᶜᶜᶜ(i, j, k, grid, u, v, w)`

at index `i, j, k` and location `ᶜᶜᶜ`.
"""
@inline ν_σᶜᶜᶜ(i, j, k, grid, ν, σᶜᶜᶜ, u, v, w) =
    @inbounds ν[i, j, k] * σᶜᶜᶜ(i, j, k, grid, u, v, w)

"""
    ν_σᶠᶠᶜ(i, j, k, grid, ν, σᶠᶠᶜ, u, v, w)

Multiply the array `ν` located at `ᶜᶜᶜ` by a function

    `σᶠᶠᶜ(i, j, k, grid, u, v, w)`

at index `i, j, k` and location `ᶠᶠᶜ`.
"""
@inline ν_σᶠᶠᶜ(i, j, k, grid, ν, σᶠᶠᶜ, u, v, w) =
    @inbounds ℑxyᶠᶠᵃ(i, j, k, grid, ν) * σᶠᶠᶜ(i, j, k, grid, u, v, w)

# These functions are analogous to the two above, but for different locations:
@inline ν_σᶠᶜᶠ(i, j, k, grid, ν, σᶠᶜᶠ, u, v, w) =
    @inbounds ℑxzᶠᵃᶠ(i, j, k, grid, ν) * σᶠᶜᶠ(i, j, k, grid, u, v, w)

@inline ν_σᶜᶠᶠ(i, j, k, grid, ν, σᶜᶠᶠ, u, v, w) =
    @inbounds ℑyzᵃᶠᶠ(i, j, k, grid, ν) * σᶜᶠᶠ(i, j, k, grid, u, v, w)

#####
##### Stress divergences
#####

# At fcc
@inline ∂x_2ν_Σ₁₁(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂xᶠᵃᵃ(i, j, k, grid, ν_σᶜᶜᶜ, diffusivities.νₑ, Σ₁₁, U.u, U.v, U.w)

@inline ∂y_2ν_Σ₁₂(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂yᵃᶜᵃ(i, j, k, grid, ν_σᶠᶠᶜ, diffusivities.νₑ, Σ₁₂, U.u, U.v, U.w)

@inline ∂z_2ν_Σ₁₃(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂zᵃᵃᶜ(i, j, k, grid, ν_σᶠᶜᶠ, diffusivities.νₑ, Σ₁₃, U.u, U.v, U.w)

# At cfc
@inline ∂x_2ν_Σ₂₁(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂xᶜᵃᵃ(i, j, k, grid, ν_σᶠᶠᶜ, diffusivities.νₑ, Σ₂₁, U.u, U.v, U.w)

@inline ∂y_2ν_Σ₂₂(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂yᵃᶠᵃ(i, j, k, grid, ν_σᶜᶜᶜ, diffusivities.νₑ, Σ₂₂, U.u, U.v, U.w)

@inline ∂z_2ν_Σ₂₃(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂zᵃᵃᶜ(i, j, k, grid, ν_σᶜᶠᶠ, diffusivities.νₑ, Σ₂₃, U.u, U.v, U.w)

# At ccf
@inline ∂x_2ν_Σ₃₁(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂xᶜᵃᵃ(i, j, k, grid, ν_σᶠᶜᶠ, diffusivities.νₑ, Σ₃₁, U.u, U.v, U.w)

@inline ∂y_2ν_Σ₃₂(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂yᵃᶜᵃ(i, j, k, grid, ν_σᶜᶠᶠ, diffusivities.νₑ, Σ₃₂, U.u, U.v, U.w)

@inline ∂z_2ν_Σ₃₃(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂zᵃᵃᶠ(i, j, k, grid, ν_σᶜᶜᶜ, diffusivities.νₑ, Σ₃₃, U.u, U.v, U.w)

"""
    ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, U, diffusivities)

Return the ``x``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₁₁) + ∂y(2 ν Σ₁₁) + ∂z(2 ν Σ₁₁)`

at the location `fcc`.
"""
@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivity, U, diffusivities) = (
      ∂x_2ν_Σ₁₁(i, j, k, grid, closure, U, diffusivities)
    + ∂y_2ν_Σ₁₂(i, j, k, grid, closure, U, diffusivities)
    + ∂z_2ν_Σ₁₃(i, j, k, grid, closure, U, diffusivities)
)

"""
    ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, U, diffusivities)

Return the ``y``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₂₁) + ∂y(2 ν Σ₂₂) + ∂z(2 ν Σ₂₂)`

at the location `ccf`.
"""
@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivity, U, diffusivities) = (
      ∂x_2ν_Σ₂₁(i, j, k, grid, closure, U, diffusivities)
    + ∂y_2ν_Σ₂₂(i, j, k, grid, closure, U, diffusivities)
    + ∂z_2ν_Σ₂₃(i, j, k, grid, closure, U, diffusivities)
)

"""
    ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, diffusivities)

Return the ``z``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₃₁) + ∂y(2 ν Σ₃₂) + ∂z(2 ν Σ₃₃)`

at the location `ccf`.
"""
@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivity, U, diffusivities) = (
      ∂x_2ν_Σ₃₁(i, j, k, grid, closure, U, diffusivities)
    + ∂y_2ν_Σ₃₂(i, j, k, grid, closure, U, diffusivities)
    + ∂z_2ν_Σ₃₃(i, j, k, grid, closure, U, diffusivities)
)
