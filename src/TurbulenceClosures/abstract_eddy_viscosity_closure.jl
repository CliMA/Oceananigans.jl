"""
    AbstractEddyViscosityClosure <: AbstractTurbulenceClosure

Abstract supertype for turbulence closures that are defined by an isotropic viscosity
and isotropic diffusivities.
"""
abstract type AbstractEddyViscosityClosure <: AbstractIsotropicDiffusivityClosure end

@inline viscosity(i, j, k, grid, closure::AbstractEddyViscosityClosure, diffusivities) = diffusivities.νₑ

