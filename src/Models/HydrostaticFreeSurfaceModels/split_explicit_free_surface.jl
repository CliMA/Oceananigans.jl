using Revise, Oceananigans, Adapt, Base
using Oceananigans.Fields
using Oceananigans.Grids
using Oceananigans.Architectures

# TODO: Potentially Change Structs before final PR
# e.g. flatten the struct, 
# auxiliary -> source / barotropic_source, 
# parameters -> gravitational_accceleration
# settings -> flattened_settings

"""
SplitExplicitFreeSurface{𝒮, 𝒫, ℰ}

# Members
state : (SplitExplicitState). The entire state for split-explicit
parameters : (NamedTuple). Parameters for timestepping split-explicit
settings : (SplitExplicitSettings). Settings for the split-explicit scheme
"""
struct SplitExplicitFreeSurface{𝒮, ℱ, 𝒫, ℰ}
    state :: 𝒮
    auxiliary :: ℱ
    parameters :: 𝒫
    settings :: ℰ
end 

# use as a trait for dispatch purposes
function SplitExplicitFreeSurface(; parameters = (; g = g_Earth),
                                    settings = SplitExplicitSettings(200),
                                    closure = nothing)

    return SplitExplicitFreeSurface(nothing, nothing, parameters, settings)
end

function FreeSurface(free_surface::SplitExplicitFreeSurface{Nothing}, velocities, arch, grid)
    return SplitExplicitFreeSurface(SplitExplicitState(grid, arch), 
                                    SplitExplicitAuxiliary(grid, arch),
                                    free_surface.parameters,
                                    free_surface.settings,
                                    )
end

function SplitExplicitFreeSurface(grid, arch; parameters = (; g = g_Earth),
    settings = SplitExplicitSettings(200),
    closure = nothing)

    sefs = SplitExplicitFreeSurface(SplitExplicitState(grid, arch),
        SplitExplicitAuxiliary(grid, arch),
        parameters,
        settings
        )

    return sefs
end

# Extend to replicate functionality: TODO delete?
function Base.getproperty(free_surface::SplitExplicitFreeSurface, sym::Symbol)
    if sym in fieldnames(SplitExplicitState)
        @assert free_surface.state isa SplitExplicitState
        return getfield(free_surface.state, sym)
    elseif sym in fieldnames(SplitExplicitAuxiliary)
        @assert free_surface.auxiliary isa SplitExplicitAuxiliary
        return getfield(free_surface.auxiliary, sym)
    elseif sym in fieldnames(SplitExplicitSettings)
        @assert free_surface.settings isa SplitExplicitSettings
        return getfield(free_surface.settings, sym)
    else
        return getfield(free_surface, sym)
    end
end

"""
SplitExplicitState{E}

# Members
`η` : (ReducedField). The instantaneous free surface 
`U` : (ReducedField). The instantaneous barotropic component of the zonal velocity 
`V` : (ReducedField). The instantaneous barotropic component of the meridional velocity
`η̅` : (ReducedField). The time-filtered free surface 
`U̅` : (ReducedField). The time-filtered barotropic component of the zonal velocity 
`V̅` : (ReducedField). The time-filtered barotropic component of the meridional velocity
"""
@Base.kwdef struct SplitExplicitState{𝒞𝒞, ℱ𝒞, 𝒞ℱ}
    η :: 𝒞𝒞
    U :: ℱ𝒞
    V :: 𝒞ℱ
    η̅ :: 𝒞𝒞
    U̅ :: ℱ𝒞
    V̅ :: 𝒞ℱ
end

function SplitExplicitState(grid::AbstractGrid, arch::AbstractArchitecture)

    η = ReducedField(Center, Center, Nothing, arch, grid; dims=3)
    η̅ = ReducedField(Center, Center, Nothing, arch, grid; dims=3)

    U = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    U̅ = ReducedField(Face, Center, Nothing, arch, grid; dims=3)

    V = ReducedField(Center, Face, Nothing, arch, grid; dims=3)
    V̅ = ReducedField(Center, Face, Nothing, arch, grid; dims=3)

    return SplitExplicitState(; η, η̅, U, U̅, V, V̅)
end

# TODO: CHANGE TO SOURCE?

"""
SplitExplicitAuxiliary{𝒞ℱ, ℱ𝒞}

# Members
`Gᵁ` : (ReducedField). Vertically integrated slow barotropic forcing function for U
`Gⱽ` : (ReducedField). Vertically integrated slow barotropic forcing function for V
`Hᶠᶜ`: (ReducedField). Depth at (Face, Center): minimum depth of neighbors
`Hᶜᶠ`: (ReducedField). Depth at (Center, Face): minimum depth of neighbors
`Hᶜᶜ`: (ReducedField). Depth at (Center, Center)
"""
@Base.kwdef struct SplitExplicitAuxiliary{𝒞ℱ, ℱ𝒞, 𝒞𝒞}
    Gᵁ :: ℱ𝒞
    Gⱽ :: 𝒞ℱ
    Hᶠᶜ:: ℱ𝒞
    Hᶜᶠ:: 𝒞ℱ
    Hᶜᶜ:: 𝒞𝒞
end

# TODO: INITIALIZE DIFFERENT DOMAIN DEPTHS from Grid
function SplitExplicitAuxiliary(grid::AbstractGrid, arch::AbstractArchitecture)

    Gᵁ = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    Gⱽ = ReducedField(Center, Face, Nothing, arch, grid; dims=3)

    Hᶠᶜ = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    Hᶜᶠ = ReducedField(Center, Face, Nothing, arch, grid; dims=3)

    Hᶜᶜ = ReducedField(Center, Center, Nothing, arch, grid; dims=3)

    return SplitExplicitAuxiliary(; Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, Hᶜᶜ)
end

"""
SplitExplicitSettings{𝒩, ℳ}

# Members
substeps: (Int)
velocity_weights : (Vector) 
free_surface_weights : (Vector)
"""
@Base.kwdef struct SplitExplicitSettings{𝒩, ℳ}
    substeps :: 𝒩
    velocity_weights :: ℳ 
    free_surface_weights :: ℳ
end

# TODO: figure out and add smart defaults here. Also make GPU-friendly (dispatch on arch?)
function SplitExplicitSettings()
    substeps = 200 # since free-surface is "substep" times faster than baroclinic part
    velocity_weights = ones(substeps) ./ substeps
    free_surface_weights = ones(substeps) ./ substeps

    return SplitExplicitSettings(substeps = substeps,
                                 velocity_weights = velocity_weights,
                                 free_surface_weights = free_surface_weights)
end

"""
SplitExplicitSettings(substeps)
"""
function SplitExplicitSettings(substeps)
    velocity_weights = ones(substeps) ./ substeps
    free_surface_weights = ones(substeps) ./ substeps

    return SplitExplicitSettings(substeps = substeps,
                                 velocity_weights = velocity_weights,
                                 free_surface_weights = free_surface_weights)
end

# Convenience Functions for grabbing free surface
free_surface(state::SplitExplicitState) = state.η
free_surface(free_surface::SplitExplicitFreeSurface) = free_surface(free_surface.state)

calculate_vertically_integrated_horizontal_velocities!(free_surface, model) = nothing

function calculate_vertically_integrated_horizontal_velocities!(free_surface::SplitExplicitFreeSurface, model)
    arch = model.architecture
    grid = model.grid
    u, v, w = model.velocities
    barotropic_corrector!(free_surface, arch, grid, u, v)
    return nothing
end

