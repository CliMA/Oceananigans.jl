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

#####
##### Multi-stage barotropic substep integrators
#####
##### `RungeKutta3Scheme` advances the fast (η, U, V) oscillator with genuine RK stages per barotropic substep.
##### It carries substep-start scratch (η⁰, U⁰, V⁰) plus the previous-stage state (ηᵖ, Uᵖ, Vᵖ), which is what
##### lets the free surface and the velocity be advanced in one kernel. The scratch is not prognostic (it is
##### recomputed at every substep), so `prognostic_state` returns `nothing`.

"""
    struct RungeKutta3Scheme

Low-storage three-stage Runge-Kutta (1/3, 1/2, 1) substep integrator for the split-explicit barotropic mode.
Within each substep the fast oscillator is advanced from the substep-start state (η⁰, U⁰, V⁰):
```math
U^{(m)} = U⁰ + γ_m (- g H ∂ η^{(m-1)} + G), \\qquad η^{(m)} = η⁰ - γ_m ∇ ⋅ U^{(m-1)},
```
with ``γ = (Δτ/3, Δτ/2, Δτ)`` and both stage right-hand sides evaluated at the previous stage.
"""
struct RungeKutta3Scheme{H, U, V, P}
    η⁰ :: H
    U⁰ :: U
    V⁰ :: V
    ηᵖ :: P
    Uᵖ :: U
    Vᵖ :: V
end

RungeKutta3Scheme() = RungeKutta3Scheme(nothing, nothing, nothing, nothing, nothing, nothing)

@inline requires_multistage(::ForwardBackwardScheme) = false
@inline requires_multistage(::RungeKutta3Scheme)     = true

function materialize_timestepper(::RungeKutta3Scheme, grid, free_surface, velocities, u_bcs, v_bcs)
    η⁰ = free_surface_displacement_field(velocities, free_surface, grid)
    ηᵖ = free_surface_displacement_field(velocities, free_surface, grid)
    U⁰ = Field{Face, Center, Nothing}(grid, boundary_conditions = u_bcs)
    V⁰ = Field{Center, Face, Nothing}(grid, boundary_conditions = v_bcs)
    Uᵖ = Field{Face, Center, Nothing}(grid, boundary_conditions = u_bcs)
    Vᵖ = Field{Center, Face, Nothing}(grid, boundary_conditions = v_bcs)
    return RungeKutta3Scheme(η⁰, U⁰, V⁰, ηᵖ, Uᵖ, Vᵖ)
end

# Per-stage substep fractions γ of Δτ. Each RK stage advances (η, U) from the substep-start state (η⁰, U⁰), with
# the tendency evaluated at the previous stage: the free surface uses η = η⁰ + γ(F − ∇·U), then the velocity uses
# the previous-stage thickness ηᵖ, U = U⁰ + γ(−gH(ηᵖ)∇ηᵖ + G). Genuine (midpoint/low-storage) RK.
@inline stage_parameters(::RungeKutta3Scheme, Δτ::FT) where FT = (Δτ / 3, Δτ / 2, Δτ)

#####
##### Timestepper extrapolations and utils
#####

function materialize_timestepper(name::Symbol, args...)
    fullname = Symbol(name, :Scheme)
    TS = getglobal(@__MODULE__, fullname)
    return materialize_timestepper(TS, args...)
end

initialize_free_surface_timestepper!(::ForwardBackwardScheme, args...) = nothing
initialize_free_surface_timestepper!(::RungeKutta3Scheme, args...) = nothing

# The functions `η★`, `U★` and `V★` represent the free surface and barotropic velocities as they enter the
# opposite update: `η★` the free surface used in the velocity update, `U★`/`V★` the transport used in the
# free-surface update (and carried into the tracer-continuity average). Forward-backward returns the current
# field in each case; the hooks exist so that a substep integrator carrying history can extrapolate instead.
@inline U★(i, j, k, grid,  ::ForwardBackwardScheme, Uᵐ)   = @inbounds Uᵐ[i, j, k]
@inline V★(i, j, k, grid,  ::ForwardBackwardScheme, Vᵐ)   = @inbounds Vᵐ[i, j, k]
@inline η★(i, j, k, grid,  ::ForwardBackwardScheme, ηᵐ⁺¹) = @inbounds ηᵐ⁺¹[i, j, k]

@inline cache_previous_free_surface!(::ForwardBackwardScheme, i, j, k, η)    = nothing
@inline   cache_previous_velocities!(::ForwardBackwardScheme, i, j, k, U, V) = nothing

#####
##### Adapt
#####

Adapt.adapt_structure(to, ts::RungeKutta3Scheme) =
    RungeKutta3Scheme(Adapt.adapt(to, ts.η⁰),
                      Adapt.adapt(to, ts.U⁰),
                      Adapt.adapt(to, ts.V⁰),
                      Adapt.adapt(to, ts.ηᵖ),
                      Adapt.adapt(to, ts.Uᵖ),
                      Adapt.adapt(to, ts.Vᵖ))

#####
##### Checkpointing
#####

prognostic_state(::ForwardBackwardScheme) = nothing
restore_prognostic_state!(restored::ForwardBackwardScheme, ::Nothing) = restored

# The Runge-Kutta scratch is recomputed at every substep, so it is not part of the prognostic state.
prognostic_state(::RungeKutta3Scheme) = nothing
restore_prognostic_state!(restored::RungeKutta3Scheme, ::Nothing) = restored
