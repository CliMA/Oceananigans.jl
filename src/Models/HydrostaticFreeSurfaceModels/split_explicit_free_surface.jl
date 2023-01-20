using Oceananigans, Adapt, Base
using Oceananigans.Fields
using Oceananigans.Grids
using Oceananigans.Architectures
using Oceananigans.AbstractOperations: Δz, GridMetricOperation
using KernelAbstractions: @index, @kernel
using Adapt

import Oceananigans.TimeSteppers: reset!
import Base.show

"""
    struct SplitExplicitFreeSurface{𝒩, 𝒮, ℱ, 𝒫 ,ℰ}

The split-explicit free surface solver.

$(TYPEDFIELDS)
"""
struct SplitExplicitFreeSurface{𝒩, 𝒮, ℱ, 𝒫 ,ℰ} <: AbstractFreeSurface{𝒩, 𝒫}
    "The instantaneous free surface (`ReducedField`)"
    η :: 𝒩
    "The entire state for the split-explicit (`SplitExplicitState`)"
    state :: 𝒮
    "Parameters for timestepping split-explicit (`NamedTuple`)"
    auxiliary :: ℱ
    "Gravitational acceleration"
    gravitational_acceleration :: 𝒫
    "Settings for the split-explicit scheme (`NamedTuple`)"
    settings :: ℰ
end

# use as a trait for dispatch purposes
SplitExplicitFreeSurface(; gravitational_acceleration = g_Earth, kwargs...) =
    SplitExplicitFreeSurface(nothing, nothing, nothing,
                             gravitational_acceleration, SplitExplicitSettings(; kwargs...))

# The new constructor is defined later on after the state, settings, auxiliary have been defined
function FreeSurface(free_surface::SplitExplicitFreeSurface, velocities, grid)
    η =  FreeSurfaceDisplacementField(velocities, free_surface, grid)

    return SplitExplicitFreeSurface(η, SplitExplicitState(grid),
                                    SplitExplicitAuxiliary(grid),
                                    free_surface.gravitational_acceleration,
                                    free_surface.settings)
end

function SplitExplicitFreeSurface(grid; gravitational_acceleration = g_Earth,
                                        settings = SplitExplicitSettings(; kwargs...))

η = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))

    return SplitExplicitFreeSurface(η,
                                    SplitExplicitState(grid),
                                    SplitExplicitAuxiliary(grid),
                                    gravitational_acceleration,
                                    settings
                                    )
end

"""
    struct SplitExplicitState{𝒞𝒞, ℱ𝒞, 𝒞ℱ}

A struct containing the state fields for the split-explicit free surface.

$(TYPEDFIELDS)
"""
Base.@kwdef struct SplitExplicitState{𝒞𝒞, ℱ𝒞, 𝒞ℱ}
    "The free surface at times at times `m`, `m-1` and `m-2`. (`ReducedField`)"
    ηᵐ   :: 𝒞𝒞
    ηᵐ⁻¹ :: 𝒞𝒞
    ηᵐ⁻² :: 𝒞𝒞
    "The instantaneous barotropic component of the zonal velocity at times `m`, `m-1` and `m-2`. (`ReducedField`)"
    U    :: ℱ𝒞
    Uᵐ⁻¹ :: ℱ𝒞
    Uᵐ⁻² :: ℱ𝒞
    "The instantaneous barotropic component of the meridional velocity at times `m`, `m-1` and `m-2`. (`ReducedField`)"
    V    :: 𝒞ℱ
    Vᵐ⁻¹ :: 𝒞ℱ
    Vᵐ⁻² :: 𝒞ℱ
    "The time-filtered free surface. (`ReducedField`)"
    η̅    :: 𝒞𝒞
    "The time-filtered barotropic component of the zonal velocity. (`ReducedField`)"
    U̅    :: ℱ𝒞
    Ũ    :: ℱ𝒞
    "The time-filtered barotropic component of the meridional velocity. (`ReducedField`)"
    V̅    :: 𝒞ℱ    
    Ṽ    :: 𝒞ℱ
end

"""
    SplitExplicitState(grid::AbstractGrid)

Return the split-explicit state. Note that `η̅` is solely used for setting the `η`
at the next substep iteration -- it essentially acts as a filter for `η`.
"""
function SplitExplicitState(grid::AbstractGrid)
    η̅ = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))

    ηᵐ   = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))
    ηᵐ⁻¹ = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))
    ηᵐ⁻² = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))
          
    U    = Field{Face, Center, Nothing}(grid)
    V    = Field{Center, Face, Nothing}(grid)

    Uᵐ⁻¹ = Field{Face, Center, Nothing}(grid)
    Vᵐ⁻¹ = Field{Center, Face, Nothing}(grid)
          
    Uᵐ⁻² = Field{Face, Center, Nothing}(grid)
    Vᵐ⁻² = Field{Center, Face, Nothing}(grid)
          
    U̅    = Field{Face, Center, Nothing}(grid)
    V̅    = Field{Center, Face, Nothing}(grid)
              
    Ũ    = Field{Face, Center, Nothing}(grid)
    Ṽ    = Field{Center, Face, Nothing}(grid)
    
    return SplitExplicitState(; ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, Uᵐ⁻¹, Uᵐ⁻², V, Vᵐ⁻¹, Vᵐ⁻², η̅, U̅, Ũ, V̅, Ṽ)
end

"""
    SplitExplicitAuxiliary{𝒞ℱ, ℱ𝒞, 𝒞𝒞}

A struct containing auxiliary fields for the split-explicit free surface.

$(TYPEDFIELDS)
"""
Base.@kwdef struct SplitExplicitAuxiliary{𝒞ℱ, ℱ𝒞, 𝒞𝒞}
    "Vertically integrated slow barotropic forcing function for `U` (`ReducedField`)"
    Gᵁ :: ℱ𝒞
    "Vertically integrated slow barotropic forcing function for `V` (`ReducedField`)"
    Gⱽ :: 𝒞ℱ
    "Depth at `(Face, Center)` (`ReducedField`)"
    Hᶠᶜ :: ℱ𝒞
    "Depth at `(Center, Face)` (`ReducedField`)"
    Hᶜᶠ :: 𝒞ℱ
    "Depth at `(Center, Center)` (`ReducedField`)"
    Hᶜᶜ :: 𝒞𝒞
end

function SplitExplicitAuxiliary(grid::AbstractGrid)

    Gᵁ = Field{Face,   Center, Nothing}(grid)
    Gⱽ = Field{Center, Face,   Nothing}(grid)

    Hᶠᶜ = Field{Face,   Center, Nothing}(grid)
    Hᶜᶠ = Field{Center, Face,   Nothing}(grid)
    Hᶜᶜ = Field{Center, Center, Nothing}(grid)

    dz = GridMetricOperation((Face, Center, Center), Δz, grid)
    sum!(Hᶠᶜ, dz)
   
    dz = GridMetricOperation((Center, Face, Center), Δz, grid)
    sum!(Hᶜᶠ, dz)

    dz = GridMetricOperation((Center, Center, Center), Δz, grid)
    sum!(Hᶜᶜ, dz)

    return SplitExplicitAuxiliary(; Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, Hᶜᶜ)
end

"""
    struct SplitExplicitSettings{𝒩, ℳ}

A struct containing settings for the split-explicit free surface.

$(TYPEDFIELDS)
"""
struct SplitExplicitSettings{𝒩, T, ℳ}
    "substeps: (`Int`)"
    substeps :: 𝒩
    "barotropic time step: (`Number`)" 
    Δτ :: T 
    "averaging_weights : (`Vector`)"
    averaging_weights :: ℳ
    "mass_flux_weights : (`Vector`)"
    mass_flux_weights :: ℳ
end

# Weights that minimize dispersion error from http://falk.ucsd.edu/roms_class/shchepetkin04.pdf (p = 2, q = 4, r = 0.18927)
@inline function averaging_shape_function(τ; p = 2, q = 4, r = 0.18927) 
    τ₀ = (p + 2) * (p + q + 2) / (p + 1) / (p + q + 1) 
    return (τ / τ₀)^p * (1 - (τ / τ₀)^q) - r * (τ / τ₀)
end

@inline averaging_cosine_function(τ) = τ >= 0.5 && τ <= 1.5 ? 1 + cos(2π * (τ - 1)) : 0.0

@inline averaging_fixed_function(τ) = 1.0

function SplitExplicitSettings(; substeps = 200, 
                                 averaging_weighting_function = averaging_fixed_function)
    
    τ = range(0.6, 2, length = 1000)

    idx = 1
    for (i, t) in enumerate(τ)
        if averaging_weighting_function(t) > 0 
            idx = i 
            break
        end
    end

    idx2 = 1
    for l in idx:1000
        idx2 = l
        averaging_weighting_function(τ[l]) <= 0 && break
    end

    τᶠ = range(0.0, τ[idx2-1], length = substeps+1)
    τᶜ = 0.5 * (τᶠ[2:end] + τᶠ[1:end-1])

    averaging_weights   = averaging_weighting_function.(τᶜ) 
    mass_flux_weights   = similar(averaging_weights)

    M = searchsortedfirst(τᶜ, 1.0) - 1

    averaging_weights ./= sum(averaging_weights)

    for i in substeps:-1:1
        mass_flux_weights[i] = 1 / M * sum(averaging_weights[i:substeps]) 
    end

    mass_flux_weights ./= sum(mass_flux_weights)

    return SplitExplicitSettings(substeps,
                                 τᶜ[2] - τᶜ[1],
                                 averaging_weights,
                                 mass_flux_weights)
end

# Convenience Functions for grabbing free surface
free_surface(free_surface::SplitExplicitFreeSurface) = free_surface.η

# extend 
@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = 0
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = 0

# convenience functor
function (sefs::SplitExplicitFreeSurface)(settings::SplitExplicitSettings)
    return SplitExplicitFreeSurface(sefs.η, sefs.state, sefs.auxiliary, sefs.gravitational_acceleration, settings)
end

Base.summary(sefs::SplitExplicitFreeSurface) = string("SplitExplicitFreeSurface with $(sefs.settings.substeps) steps")

Base.show(io::IO, sefs::SplitExplicitFreeSurface) = print(io, "$(summary(sefs))\n")

function reset!(sefs::SplitExplicitFreeSurface)
    for name in propertynames(sefs.state)
        var = getproperty(sefs.state, name)
        fill!(var, 0.0)
    end
end

# Adapt
Adapt.adapt_structure(to, free_surface::SplitExplicitFreeSurface) =
    SplitExplicitFreeSurface(Adapt.adapt(to, free_surface.η), nothing, nothing,
                             free_surface.gravitational_acceleration, nothing)
