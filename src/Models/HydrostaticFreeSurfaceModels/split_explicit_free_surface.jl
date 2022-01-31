using Oceananigans, Adapt, Base
using Oceananigans.Fields
using Oceananigans.Grids
using Oceananigans.Architectures
using Oceananigans.AbstractOperations: Δz, GridMetricOperation
using KernelAbstractions: @index, @kernel
using Adapt

import Base.show

"""
SplitExplicitFreeSurface{𝒮, 𝒫, ℰ}

# Members
`η` : (ReducedField). The instantaneous free surface 
`state` : (SplitExplicitState). The entire state for split-explicit
`gravitational_acceleration` : (NamedTuple). Parameters for timestepping split-explicit
`settings` : (SplitExplicitSettings). Settings for the split-explicit scheme

$(TYPEDFIELDS)
"""
struct SplitExplicitFreeSurface{𝒩, 𝒮, ℱ, 𝒫 ,ℰ}
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
function SplitExplicitFreeSurface(; gravitational_acceleration = g_Earth, substeps = 200)

    return SplitExplicitFreeSurface(nothing, nothing, nothing,
                                    gravitational_acceleration, SplitExplicitSettings(substeps))
end

# The new constructor is defined later on after the state, settings, auxiliary have been defined
function FreeSurface(free_surface::SplitExplicitFreeSurface, velocities, grid)
    η =  Field{Center, Center, Nothing}(grid)

    return SplitExplicitFreeSurface(η, SplitExplicitState(grid),
                                    SplitExplicitAuxiliary(grid),
                                    free_surface.gravitational_acceleration,
                                    free_surface.settings)
end

function SplitExplicitFreeSurface(grid; gravitational_acceleration = g_Earth,
    settings = SplitExplicitSettings(200))
    η =  Field{Center, Center, Nothing}(grid)
    sefs = SplitExplicitFreeSurface(η, SplitExplicitState(grid),
                                    SplitExplicitAuxiliary(grid),
                                    gravitational_acceleration,
                                    settings
                                    )

    return sefs
end

"""
    struct SplitExplicitState{𝒞𝒞, ℱ𝒞, 𝒞ℱ}

A struct containing the state fields for the split-explicit free surface.

$(TYPEDFIELDS)
"""
Base.@kwdef struct SplitExplicitState{𝒞𝒞, ℱ𝒞, 𝒞ℱ}
    "The instantaneous barotropic component of the zonal velocity. (`ReducedField`)"
    U :: ℱ𝒞
    "The instantaneous barotropic component of the meridional velocity. (`ReducedField`)"
    V :: 𝒞ℱ
    "The time-filtered free surface. (`ReducedField`)"
    η̅ :: 𝒞𝒞
    "The time-filtered barotropic component of the zonal velocity. (`ReducedField`)"
    U̅ :: ℱ𝒞
    "The time-filtered barotropic component of the meridional velocity. (`ReducedField`)"
    V̅ :: 𝒞ℱ
end

"""
    SplitExplicitState(grid::AbstractGrid)

Return the split-explicit state. Note that `η̅` is solely used for setting the `η`
at the next substep iteration -- it essentially acts as a filter for `η`.
"""
function SplitExplicitState(grid::AbstractGrid)
    η̅ = Field{Center, Center, Nothing}(grid)

    U = Field{Face, Center, Nothing}(grid)
    U̅ = Field{Face, Center, Nothing}(grid)

    V = Field{Center, Face, Nothing}(grid)
    V̅ = Field{Center, Face, Nothing}(grid)

    return SplitExplicitState(; U, V, η̅, U̅, V̅)
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

    Gᵁ = Field{Face,Center,Nothing}(grid)
    Gⱽ = Field{Center,Face,Nothing}(grid)

    Hᶠᶜ = Field{Face,Center,Nothing}(grid)
    Hᶜᶠ = Field{Center,Face,Nothing}(grid)
    Hᶜᶜ = Field{Center,Center,Nothing}(grid)

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
struct SplitExplicitSettings{𝒩, ℳ}
    "substeps: (`Int`)"
    substeps :: 𝒩
    "velocity_weights : (`Vector`)"
    velocity_weights :: ℳ
    "free_surface_weights : (`Vector`)"
    free_surface_weights :: ℳ
end

function SplitExplicitSettings(; substeps = 200, velocity_weights = nothing, free_surface_weights = nothing)
    velocity_weights = Tuple(ones(substeps) ./ substeps)
    free_surface_weights = Tuple(ones(substeps) ./ substeps)

    return SplitExplicitSettings(substeps,
        velocity_weights,
        free_surface_weights)
end

function SplitExplicitSettings(substeps)
    velocity_weights = Tuple(ones(substeps) ./ substeps)
    free_surface_weights = Tuple(ones(substeps) ./ substeps)

    return SplitExplicitSettings(substeps = substeps,
        velocity_weights = velocity_weights,
        free_surface_weights = free_surface_weights)
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

# Adapt
Adapt.adapt_structure(to, free_surface::SplitExplicitFreeSurface) =
    SplitExplicitFreeSurface(Adapt.adapt(to, free_surface.η), nothing, nothing, free_surface.gravitational_acceleration,
        nothing)
