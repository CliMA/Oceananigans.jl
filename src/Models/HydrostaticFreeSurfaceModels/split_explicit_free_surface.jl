using Oceananigans, Adapt, Base
using Oceananigans.Fields
using Oceananigans.Grids
using Oceananigans.Architectures
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶠᶜ, Δzᶠᶜᶜ 
using KernelAbstractions: @index, @kernel
import Base.show

# TODO: Potentially Change Structs before final PR
# e.g. flatten the struct, 
# auxiliary -> source / barotropic_source, 
# gravitational_acceleration
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
    gravitational_acceleration :: 𝒫
    settings :: ℰ
end 

# use as a trait for dispatch purposes
function SplitExplicitFreeSurface(; gravitational_acceleration = g_Earth,
                                    substeps = 200)

    return SplitExplicitFreeSurface(nothing, nothing, gravitational_acceleration, SplitExplicitSettings(substeps))
end

function FreeSurface(free_surface::SplitExplicitFreeSurface{Nothing}, velocities, arch, grid)
    return SplitExplicitFreeSurface(SplitExplicitState(grid, arch), 
                                    SplitExplicitAuxiliary(grid, arch),
                                    free_surface.gravitational_acceleration, 
                                    free_surface.settings)
end

function SplitExplicitFreeSurface(grid, arch; gravitational_acceleration = g_Earth,
                                  settings = SplitExplicitSettings(200))

    sefs = SplitExplicitFreeSurface(SplitExplicitState(grid, arch),
        SplitExplicitAuxiliary(grid, arch),
        gravitational_acceleration,
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

function SplitExplicitAuxiliary(grid::AbstractGrid, arch::AbstractArchitecture)

    Gᵁ = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    Gⱽ = ReducedField(Center, Face, Nothing, arch, grid; dims=3)

    Hᶠᶜ = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    Hᶜᶠ = ReducedField(Center, Face, Nothing, arch, grid; dims=3)

    Hᶜᶜ = ReducedField(Center, Center, Nothing, arch, grid; dims=3)

    event = launch!(arch, grid, :xy, initialize_vertical_depths_kernel!, 
                    Hᶠᶜ, Hᶜᶠ, Hᶜᶜ, grid, dependencies=Event(device(arch)))

    wait(device(arch), event)
    
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

@kernel function initialize_vertical_depths_kernel!(Hᶠᶜ, Hᶜᶠ, Hᶜᶜ, grid)
    i, j = @index(Global, NTuple)

    @inbounds begin
        Hᶠᶜ[i, j, 1] = 0
        Hᶜᶠ[i, j, 1] = 0
        Hᶜᶜ[i, j, 1] = 0

        @unroll for k in 1:grid.Nz
            Hᶠᶜ[i, j, 1] += Δzᶠᶜᶜ(i, j, k, grid)
            Hᶜᶠ[i, j, 1] += Δzᶜᶠᶜ(i, j, k, grid)
            Hᶜᶜ[i, j, 1] += Δzᶜᶜᶜ(i, j, k, grid)
        end
    end
end 
