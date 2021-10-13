using Oceananigans.Grids: AbstractGrid
using Oceananigans.Architectures: device
using Oceananigans.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Solvers: solve!
using Oceananigans.Fields
using Oceananigans.Utils: prettytime

using Adapt
using KernelAbstractions: NoneEvent

"""
SplitExplicitFreeSurface{𝒮, 𝒫, ℰ}

# Members
state : (SplitExplicitState). The entire state for split-explicit
parameters : (named tuple). Parameters for timestepping split-explicit
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

"""
SplitExplicitState{E}

# Members
`η` : (ReducedField). The instantaneous free surface 
`U` : (ReducedField). The instantaneous barotropic component of the zonal velocity 
`V` : (ReducedField). The instantaneous batropic component of the meridional velocity
`η̅` : (ReducedField). The time-filtered free surface 
`U̅` : (ReducedField). The time-filtered barotropic component of the zonal velocity 
`V̅` : (ReducedField). The time-filtered batropic component of the meridional velocity
"""
@Base.kwdef struct SplitExplicitState{𝒮}
    η :: 𝒮
    U :: 𝒮
    V :: 𝒮
    η̅ :: 𝒮
    U̅ :: 𝒮
    V̅ :: 𝒮
end

# TODO: given the grid construct the members of the struct
function SplitExplicitState(grid)
    # make split-explicit stuff here
    return nothing
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

#TODO: figure out and add smart defualts here. Also make GPU-friendly
function SplitExplicitSettings()
    substeps = 200 # since free-surface is substep times faster than baroclinic part
    velocity_weights = ones(substeps) ./ substeps
    free_surface_weights = ones(substeps) ./ substeps

    return SplitExplicitSettings(substeps = substeps,
                                 velocity_weights = velocity_weights,
                                 free_surface_weights = free_surface_weights)
end