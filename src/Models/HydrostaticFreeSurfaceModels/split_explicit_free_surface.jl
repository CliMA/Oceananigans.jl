using Oceananigans, Adapt, Base
using Oceananigans.Fields
using Oceananigans.Grids
using Oceananigans.Architectures
using Oceananigans.AbstractOperations: Δz, GridMetricOperation
using KernelAbstractions: @index, @kernel
using Adapt
import Base.show

# TODO: Potentially Change Structs before final PR
# e.g. flatten the struct, 
# auxiliary -> source / barotropic_source, 
# gravitational_acceleration
# settings -> flattened_settings

"""
SplitExplicitFreeSurface{𝒮, 𝒫, ℰ}

# Members
`η` : (ReducedField). The instantaneous free surface 
`state` : (SplitExplicitState). The entire state for split-explicit
`gravitational_acceleration` : (NamedTuple). Parameters for timestepping split-explicit
`settings` : (SplitExplicitSettings). Settings for the split-explicit scheme
"""
struct SplitExplicitFreeSurface{𝒩,𝒮,ℱ,𝒫,ℰ}
    η::𝒩
    state::𝒮
    auxiliary::ℱ
    gravitational_acceleration::𝒫
    settings::ℰ
end

# use as a trait for dispatch purposes
function SplitExplicitFreeSurface(; gravitational_acceleration = g_Earth,
    substeps = 200)

    return SplitExplicitFreeSurface(nothing, nothing, nothing, gravitational_acceleration, SplitExplicitSettings(substeps))
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
SplitExplicitState{E}

# Members
`U` : (ReducedField). The instantaneous barotropic component of the zonal velocity 
`V` : (ReducedField). The instantaneous barotropic component of the meridional velocity
`η̅` : (ReducedField). The time-filtered free surface 
`U̅` : (ReducedField). The time-filtered barotropic component of the zonal velocity 
`V̅` : (ReducedField). The time-filtered barotropic component of the meridional velocity
"""
Base.@kwdef struct SplitExplicitState{𝒞𝒞,ℱ𝒞,𝒞ℱ}
    U::ℱ𝒞
    V::𝒞ℱ
    η̅::𝒞𝒞
    U̅::ℱ𝒞
    V̅::𝒞ℱ
end

# η̅ is solely used for setting the eta at the next substep iteration
# it essentially acts as a filter for η

function SplitExplicitState(grid::AbstractGrid)

    η̅ = Field{Center, Center, Nothing}(grid)

    U = Field{Face, Center, Nothing}(grid)
    U̅ = Field{Face, Center, Nothing}(grid)

    V = Field{Center, Face, Nothing}(grid)
    V̅ = Field{Center, Face, Nothing}(grid)

    return SplitExplicitState(; U, V, η̅, U̅, V̅)
end

# TODO: CHANGE TO SOURCE?

"""
SplitExplicitAuxiliary{𝒞ℱ, ℱ𝒞}

# Members
`Gᵁ` : (ReducedField). Vertically integrated slow barotropic forcing function for U
`Gⱽ` : (ReducedField). Vertically integrated slow barotropic forcing function for V
`Hᶠᶜ`: (ReducedField). Depth at (Face, Center)
`Hᶜᶠ`: (ReducedField). Depth at (Center, Face)
`Hᶜᶜ`: (ReducedField). Depth at (Center, Center)
"""
Base.@kwdef struct SplitExplicitAuxiliary{𝒞ℱ,ℱ𝒞,𝒞𝒞}
    Gᵁ::ℱ𝒞
    Gⱽ::𝒞ℱ
    Hᶠᶜ::ℱ𝒞
    Hᶜᶠ::𝒞ℱ
    Hᶜᶜ::𝒞𝒞
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
SplitExplicitSettings{𝒩, ℳ}

# Members
substeps: (Int)
velocity_weights : (Vector) 
free_surface_weights : (Vector)
"""
struct SplitExplicitSettings{𝒩,ℳ}
    substeps::𝒩
    velocity_weights::ℳ
    free_surface_weights::ℳ
end

function SplitExplicitSettings(; substeps = 200, velocity_weights = nothing, free_surface_weights = nothing)
    velocity_weights = Tuple(ones(substeps) ./ substeps)
    free_surface_weights = Tuple(ones(substeps) ./ substeps)

    return SplitExplicitSettings(substeps,
        velocity_weights,
        free_surface_weights)
end

"""
SplitExplicitSettings(substeps)
"""
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
