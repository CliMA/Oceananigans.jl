"""
    AbstractEddyViscosityClosure <: AbstractTurbulenceClosure

Abstract supertype for turbulence closures that are defined by an isotropic viscosity
and isotropic diffusivities.
"""
abstract type AbstractEddyViscosityClosure <: AbstractTurbulenceClosure end

#####
##### Viscosity-stress products
#####

"""
    ν_σᶜᶜᶜ(i, j, k, grid, ν, σᶜᶜᶜ, u, v, w)

Multiply the array `ν` located at `ᶜᶜᶜ` by a function

    `σᶜᶜᶜ(i, j, k, grid, u, v, w)`

at index `i, j, k` and location `ᶜᶜᶜ`.
"""
@inline ν_σᶜᶜᶜ(i, j, k, grid, ν, σᶜᶜᶜ, u, v, w) = @inbounds ν[i, j, k] * σᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline ν_σᶠᶠᶜ(i, j, k, grid, ν, σᶠᶠᶜ, u, v, w) = @inbounds ℑxyᶠᶠᵃ(i, j, k, grid, ν) * σᶠᶠᶜ(i, j, k, grid, u, v, w)
@inline ν_σᶠᶜᶠ(i, j, k, grid, ν, σᶠᶜᶠ, u, v, w) = @inbounds ℑxzᶠᵃᶠ(i, j, k, grid, ν) * σᶠᶜᶠ(i, j, k, grid, u, v, w)
@inline ν_σᶜᶠᶠ(i, j, k, grid, ν, σᶜᶠᶠ, u, v, w) = @inbounds ℑyzᵃᶠᶠ(i, j, k, grid, ν) * σᶜᶠᶠ(i, j, k, grid, u, v, w)

#####
##### Stress divergences
#####

viscous_flux_ux(i, j, k, grid, clock, closure::AbstractEddyViscosityClosure, U, diffusivities) = 2 * ν_σᶜᶜᶜ(i, j, k, grid, diffusivities.νₑ, Σ₁₁, U.u, U.v, U.w)
viscous_flux_uy(i, j, k, grid, clock, closure::AbstractEddyViscosityClosure, U, diffusivities) = 2 * ν_σᶠᶠᶜ(i, j, k, grid, diffusivities.νₑ, Σ₁₂, U.u, U.v, U.w)
viscous_flux_uz(i, j, k, grid, clock, closure::AbstractEddyViscosityClosure, U, diffusivities) = 2 * ν_σᶠᶜᶠ(i, j, k, grid, diffusivities.νₑ, Σ₁₃, U.u, U.v, U.w)

viscous_flux_vx(i, j, k, grid, clock, closure::AbstractEddyViscosityClosure, U, diffusivities) = 2 * ν_σᶠᶠᶜ(i, j, k, grid, diffusivities.νₑ, Σ₂₁, U.u, U.v, U.w)
viscous_flux_vy(i, j, k, grid, clock, closure::AbstractEddyViscosityClosure, U, diffusivities) = 2 * ν_σᶜᶜᶜ(i, j, k, grid, diffusivities.νₑ, Σ₂₂, U.u, U.v, U.w)
viscous_flux_vz(i, j, k, grid, clock, closure::AbstractEddyViscosityClosure, U, diffusivities) = 2 * ν_σᶜᶠᶠ(i, j, k, grid, diffusivities.νₑ, Σ₂₃, U.u, U.v, U.w)

viscous_flux_wx(i, j, k, grid, clock, closure::AbstractEddyViscosityClosure, U, diffusivities) = 2 * ν_σᶠᶜᶠ(i, j, k, grid, diffusivities.νₑ, Σ₃₁, U.u, U.v, U.w)
viscous_flux_wy(i, j, k, grid, clock, closure::AbstractEddyViscosityClosure, U, diffusivities) = 2 * ν_σᶜᶠᶠ(i, j, k, grid, diffusivities.νₑ, Σ₃₂, U.u, U.v, U.w)
viscous_flux_wz(i, j, k, grid, clock, closure::AbstractEddyViscosityClosure, U, diffusivities) = 2 * ν_σᶜᶜᶜ(i, j, k, grid, diffusivities.νₑ, Σ₃₃, U.u, U.v, U.w)
