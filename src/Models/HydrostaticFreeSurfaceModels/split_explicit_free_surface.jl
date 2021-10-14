using Oceananigans.Fields
using Adapt

"""
SplitExplicitFreeSurface{𝒮, 𝒫, ℰ}

# Members
state : (SplitExplicitState). The entire state for split-explicit
parameters : (NamedTuple). Parameters for timestepping split-explicit
settings : (SplitExplicitSettings). Settings for the split-explicit scheme
"""
@Base.kwdef struct SplitExplicitFreeSurface{E, G, M}
    state :: 𝒮
    parameters :: 𝒫
    settings :: ℰ
end

# use as a trait for dispatch purposes
function SplitExplicitFreeSurface()
    return SplitExplicitFreeSurface(nothing, nothing, nothing)
end

# automatically construct default
function SplitExplicitFreeSurface(grid, arch)
    return SplitExplicitFreeSurface(state = SplitExplicitState(grid, arch), 
                                    parameters = (; g = g_Earth), 
                                    settings = SplitExplicitSettings(),)
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
@Base.kwdef struct SplitExplicitState{𝒮}
    η :: 𝒮
    U :: 𝒮
    V :: 𝒮
    η̅ :: 𝒮
    U̅ :: 𝒮
    V̅ :: 𝒮
end

function SplitExplicitState(grid, arch)
    η = ReducedField(Center, Center, Nothing, arch, grid; dims=3)
    η̅ = ReducedField(Center, Center, Nothing, arch, grid; dims=3)

    U = ReducedField(Face, Center, Nothing, arch, grid; dims=3)
    U̅ = ReducedField(Face, Center, Nothing, arch, grid; dims=3)

    V = ReducedField(Center, Face, Nothing, arch, grid; dims=3)
    V̅ = ReducedField(Center, Face, Nothing, arch, grid; dims=3)

    return SplitExplicitState(η = η, η̅ = η̅, U = U, U̅ = U̅, V = V, V̅ = V̅)
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

#TODO: figure out and add smart defaults here. Also make GPU-friendly (dispatch on arch?)
function SplitExplicitSettings()
    substeps = 200 # since free-surface is "substep" times faster than baroclinic part
    velocity_weights = ones(substeps) ./ substeps
    free_surface_weights = ones(substeps) ./ substeps

    return SplitExplicitSettings(substeps = substeps,
                                 velocity_weights = velocity_weights,
                                 free_surface_weights = free_surface_weights)
end