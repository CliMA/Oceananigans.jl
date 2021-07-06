"""
    AbstractEddyViscosityClosure <: AbstractTurbulenceClosure

Abstract supertype for turbulence closures that are defined by an isotropic viscosity
and isotropic diffusivities.
"""
abstract type AbstractEddyViscosityClosure{TD} <: AbstractIsotropicDiffusivity{TD} end

@inline viscosity(closure::AbstractEddyViscosityClosure, diffusivities, args...) = diffusivities.νₑ
@inline diffusivity(::AbstractEddyViscosityClosure, ::Val{tracer_index}, diffusivities, args...) where tracer_index = diffusivities.κₑ[tracer_index]
    
