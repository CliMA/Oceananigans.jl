using Adapt: Adapt

import Oceananigans: prognostic_state, restore_prognostic_state!

"""
    struct ForwardBackwardScheme

A timestepping scheme used for substepping in the split-explicit free surface solver.

The equations are evolved as follows:
```math
\\begin{gather}
η^{m+1} = η^m - Δτ (∂_x U^m + ∂_y V^m), \\\\
U^{m+1} = U^m - Δτ (∂_x η^{m+1} - G^U), \\\\
V^{m+1} = V^m - Δτ (∂_y η^{m+1} - G^V).
\\end{gather}
```
"""
struct ForwardBackwardScheme end

materialize_timestepper(::ForwardBackwardScheme, grid, args...) = ForwardBackwardScheme()

#####
##### Timestepper extrapolations and utils
#####

function materialize_timestepper(name::Symbol, args...)
    fullname = Symbol(name, :Scheme)
    TS = getglobal(@__MODULE__, fullname)
    return materialize_timestepper(TS, args...)
end

initialize_free_surface_timestepper!(::ForwardBackwardScheme, args...) = nothing

# The functions `η★` `U★` and `V★` represent the value of free surface, barotropic zonal and meridional velocity at time step m+1/2
@inline U★(i, j, k, grid,  ::ForwardBackwardScheme, Uᵐ)   = @inbounds Uᵐ[i, j, k]
@inline η★(i, j, k, grid,  ::ForwardBackwardScheme, ηᵐ⁺¹) = @inbounds ηᵐ⁺¹[i, j, k]

@inline   cache_previous_velocities!(::ForwardBackwardScheme, i, j, k, U, V) = nothing
@inline cache_previous_free_surface!(::ForwardBackwardScheme, i, j, k, η)    = nothing

#####
##### Checkpointing
#####

prognostic_state(::ForwardBackwardScheme) = nothing
restore_prognostic_state!(ts::ForwardBackwardScheme, ::Nothing) = ts
