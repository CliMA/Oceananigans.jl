abstract type AbstractIsotropicDiffusivityClosure <: AbstractTurbulenceClosure end

#####
##### Viscosity-stress products
#####

"""
    ν_σᶜᶜᶜ(i, j, k, grid, ν, σᶜᶜᶜ, u, v, w)

Multiply the array `ν` located at `ᶜᶜᶜ` by a function

    `σᶜᶜᶜ(i, j, k, grid, u, v, w)`

at index `i, j, k` and location `ᶜᶜᶜ`.
"""
@inline ν_σᶜᶜᶜ(i, j, k, grid, clock, ν, σᶜᶜᶜ, u, v, w) = @inbounds νᶜᶜᶜ(i, j, k, grid, clock, ν) * σᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline ν_σᶠᶠᶜ(i, j, k, grid, clock, ν, σᶠᶠᶜ, u, v, w) = @inbounds νᶠᶠᶜ(i, j, k, grid, clock, ν) * σᶠᶠᶜ(i, j, k, grid, u, v, w)
@inline ν_σᶠᶜᶠ(i, j, k, grid, clock, ν, σᶠᶜᶠ, u, v, w) = @inbounds νᶠᶜᶠ(i, j, k, grid, clock, ν) * σᶠᶜᶠ(i, j, k, grid, u, v, w)
@inline ν_σᶜᶠᶠ(i, j, k, grid, clock, ν, σᶜᶠᶠ, u, v, w) = @inbounds νᶜᶠᶠ(i, j, k, grid, clock, ν) * σᶜᶠᶠ(i, j, k, grid, u, v, w)

#####
##### Stress divergences
#####

"""
    viscosity(i, j, k, grid, closure, diffusivities)

Returns viscosity either as a number at `i, j, k`, or as a function or array.
"""
function viscosity end 

@inline viscous_flux_ux(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivityClosure, U, diffusivities) = 2 * ν_σᶜᶜᶜ(i, j, k, grid, closure, viscosity(i, j, k, grid, closure, diffusivities), Σ₁₁, U.u, U.v, U.w)
@inline viscous_flux_uy(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivityClosure, U, diffusivities) = 2 * ν_σᶠᶠᶜ(i, j, k, grid, closure, viscosity(i, j, k, grid, closure, diffusivities), Σ₁₂, U.u, U.v, U.w)
@inline viscous_flux_uz(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivityClosure, U, diffusivities) = 2 * ν_σᶠᶜᶠ(i, j, k, grid, closure, viscosity(i, j, k, grid, closure, diffusivities), Σ₁₃, U.u, U.v, U.w)

@inline viscous_flux_vx(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivityClosure, U, diffusivities) = 2 * ν_σᶠᶠᶜ(i, j, k, grid, closure, viscosity(i, j, k, grid, closure, diffusivities), Σ₂₁, U.u, U.v, U.w)
@inline viscous_flux_vy(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivityClosure, U, diffusivities) = 2 * ν_σᶜᶜᶜ(i, j, k, grid, closure, viscosity(i, j, k, grid, closure, diffusivities), Σ₂₂, U.u, U.v, U.w)
@inline viscous_flux_vz(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivityClosure, U, diffusivities) = 2 * ν_σᶜᶠᶠ(i, j, k, grid, closure, viscosity(i, j, k, grid, closure, diffusivities), Σ₂₃, U.u, U.v, U.w)

@inline viscous_flux_wx(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivityClosure, U, diffusivities) = 2 * ν_σᶠᶜᶠ(i, j, k, grid, closure, viscosity(i, j, k, grid, closure, diffusivities), Σ₃₁, U.u, U.v, U.w)
@inline viscous_flux_wy(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivityClosure, U, diffusivities) = 2 * ν_σᶜᶠᶠ(i, j, k, grid, closure, viscosity(i, j, k, grid, closure, diffusivities), Σ₃₂, U.u, U.v, U.w)
@inline viscous_flux_wz(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivityClosure, U, diffusivities) = 2 * ν_σᶜᶜᶜ(i, j, k, grid, closure, viscosity(i, j, k, grid, closure, diffusivities), Σ₃₃, U.u, U.v, U.w)

#####
##### Diffusive fluxes
#####

"""
    diffusivity(i, j, k, grid, closure, diffusivities, ::Val{tracer_index}) where tracer_index

Returns tracer diffusivity for tracer_index either as a number at `i, j, k`, or as a function or array.
"""
function diffusivity end 

@inline function diffusive_flux_x(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivity, c, ::Val{tracer_index}, diffusivities, args...) where tracer_index
    κ = diffusivity(i, j, k, grid, closure, diffusivities, Val(tracer_index))
    return diffusive_flux_x(i, j, k, grid, clock, κ, c)
end

@inline function diffusive_flux_x(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivity, c, ::Val{tracer_index}, diffusivities, args...) where tracer_index
    κ = diffusivity(i, j, k, grid, closure, diffusivities, Val(tracer_index))
    return diffusive_flux_x(i, j, k, grid, clock, κ, c)
end

@inline function diffusive_flux_x(i, j, k, grid, clock, closure::AbstractIsotropicDiffusivity, c, ::Val{tracer_index}, diffusivities, args...) where tracer_index
    κ = diffusivity(i, j, k, grid, closure, diffusivities, Val(tracer_index))
    return diffusive_flux_x(i, j, k, grid, clock, κ, c)
end

