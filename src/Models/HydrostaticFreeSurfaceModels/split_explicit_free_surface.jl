using Revise, Oceananigans, Adapt, Base
using Oceananigans.Fields
using Oceananigans.Grids
using Oceananigans.Architectures

"""
SplitExplicitFreeSurface{𝒮, 𝒫, ℰ}

# Members
state : (SplitExplicitState). The entire state for split-explicit
parameters : (NamedTuple). Parameters for timestepping split-explicit
settings : (SplitExplicitSettings). Settings for the split-explicit scheme
"""
@Base.kwdef struct SplitExplicitFreeSurface{𝒮, ℱ, 𝒫, ℰ}
    state :: 𝒮
    forcing :: ℱ
    parameters :: 𝒫
    settings :: ℰ
end


# use as a trait for dispatch purposes
function SplitExplicitFreeSurface()
    return SplitExplicitFreeSurface(nothing, nothing, nothing, nothing)
end

# automatically construct default
function SplitExplicitFreeSurface(grid::AbstractGrid, arch::AbstractArchitecture)
    return SplitExplicitFreeSurface(state = SplitExplicitState(grid, arch), 
                                    forcing = SplitExplicitForcing(grid, arch),
                                    parameters = (; g = g_Earth), 
                                    settings = SplitExplicitSettings(),)
end

# Extend to replicate functionality: TODO delete?
function Base.getproperty(free_surface::SplitExplicitFreeSurface, sym::Symbol)
    if sym in fieldnames(SplitExplicitState)
        @assert free_surface.state isa SplitExplicitState
        return getfield(free_surface.state, sym)
    elseif sym in fieldnames(SplitExplicitForcing)
        @assert free_surface.forcing isa SplitExplicitForcing
        return getfield(free_surface.forcing, sym)
    else
        return getfield(free_surface, sym)
    end
end


free_surface(state::SplitExplicitState) = state.η
free_surface(free_surface::SplitExplicitFreeSurface) = free_surface(free_surface.state)


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


"""
SplitExplicitForcing{𝒞ℱ, ℱ𝒞}

# Members
`Gᵁ` : (ReducedField). Vertically integrated slow barotropic forcing function for U
`Gⱽ` : (ReducedField). Vertically integrated slow barotropic forcing function for V
"""
@Base.kwdef struct SplitExplicitForcing{𝒞ℱ, ℱ𝒞}
    Gᵁ :: 𝒞ℱ
    Gⱽ :: ℱ𝒞
end

function SplitExplicitForcing(grid::AbstractGrid, arch::AbstractArchitecture)

    Gᵁ = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    Gⱽ = ReducedField(Center, Face, Nothing, arch, grid; dims=3)

    return SplitExplicitForcing(; Gᵁ, Gⱽ)
end

"""
SplitExplicitSettings{𝒩, ℳ}

# Members
substeps: (Int)
velocity_weights :: (Vector) 
free_surface_weights :: (Vector)
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