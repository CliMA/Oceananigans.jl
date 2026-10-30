using Adapt: Adapt

import Oceananigans: prognostic_state, restore_prognostic_state!

"""
    struct ForwardBackwardScheme

A timestepping scheme used for substepping in the split-explicit free surface solver.

The equations are evolved as follows:
```math
\\begin{gather}
U^{m+1} = U^m - Δτ (∂_x η^m - G^U), \\\\
V^{m+1} = V^m - Δτ (∂_y η^m - G^V), \\\\
η^{m+1} = η^m - Δτ (∂_x U^{m+1} + ∂_y V^{m+1}).
\\end{gather}
```
"""
struct ForwardBackwardScheme end

materialize_timestepper(::ForwardBackwardScheme, grid, args...) = ForwardBackwardScheme()

"""
    struct CoriolisForwardBackwardScheme

A `ForwardBackwardScheme` that also advances the barotropic Coriolis force inside the substeps.

The slow tendencies ``G^U`` and ``G^V`` hold the Coriolis force of the baroclinic reference transports
``U^r`` and ``V^r`` frozen over the whole baroclinic step. This scheme adds the Coriolis force of the
deviation from that reference, alternating the order of the two components (forward-backward in ``f``):
```math
\\begin{gather}
U^{m+1} = U^m - Δτ (g H ∂_x η^m - G^U - f (V^m - V^r)), \\\\
V^{m+1} = V^m - Δτ (g H ∂_y η^m - G^V + f (U^{m+1} - U^r)), \\\\
η^{m+1} = η^m - Δτ (∂_x U^{m+1} + ∂_y V^{m+1}).
\\end{gather}
```
For `QuasiAdamsBashforth2TimeStepper`, ``U^r = (3/2 + χ) U^n - (1/2 + χ) U^{n-1}``; for
`SplitRungeKuttaTimeStepper`, ``U^r`` is the barotropic transport at the start of the stage.
"""
struct CoriolisForwardBackwardScheme{R, P}
    reference_transports :: R
    previous_transports :: P
end

CoriolisForwardBackwardScheme() = CoriolisForwardBackwardScheme(nothing, nothing)

const ForwardBackwardSchemes = Union{ForwardBackwardScheme, CoriolisForwardBackwardScheme}

function materialize_timestepper(::CoriolisForwardBackwardScheme, grid, free_surface, velocities, U_bcs, V_bcs)
    Uʳ = Field{Face, Center, Nothing}(grid, boundary_conditions = U_bcs)
    Vʳ = Field{Center, Face, Nothing}(grid, boundary_conditions = V_bcs)
    U⁻ = Field{Face, Center, Nothing}(grid, boundary_conditions = U_bcs)
    V⁻ = Field{Center, Face, Nothing}(grid, boundary_conditions = V_bcs)
    return CoriolisForwardBackwardScheme((U = Uʳ, V = Vʳ), (U = U⁻, V = V⁻))
end

Adapt.adapt_structure(to, ts::CoriolisForwardBackwardScheme) =
    CoriolisForwardBackwardScheme(Adapt.adapt(to, ts.reference_transports), nothing)

#####
##### Timestepper extrapolations and utils
#####

function materialize_timestepper(name::Symbol, args...)
    fullname = Symbol(name, :Scheme)
    TS = getglobal(@__MODULE__, fullname)
    return materialize_timestepper(TS, args...)
end

initialize_free_surface_timestepper!(::ForwardBackwardSchemes, args...) = nothing

# The functions `η★` `U★` and `V★` represent the value of free surface, barotropic zonal and meridional velocity at time step m+1/2
@inline U★(i, j, k, grid,  ::ForwardBackwardSchemes, Uᵐ)   = @inbounds Uᵐ[i, j, k]
@inline η★(i, j, k, grid,  ::ForwardBackwardSchemes, ηᵐ⁺¹) = @inbounds ηᵐ⁺¹[i, j, k]

@inline cache_previous_free_surface!(::ForwardBackwardSchemes, i, j, k, η)    = nothing
@inline   cache_previous_velocities!(::ForwardBackwardSchemes, i, j, k, U, V) = nothing

#####
##### Checkpointing
#####

prognostic_state(::ForwardBackwardScheme) = nothing
restore_prognostic_state!(restored::ForwardBackwardScheme, ::Nothing) = restored

prognostic_state(ts::CoriolisForwardBackwardScheme) = (; previous_transports = prognostic_state(ts.previous_transports))

function restore_prognostic_state!(restored::CoriolisForwardBackwardScheme, from)
    restore_prognostic_state!(restored.previous_transports, from.previous_transports)
    return restored
end

restore_prognostic_state!(restored::CoriolisForwardBackwardScheme, ::Nothing) = restored
