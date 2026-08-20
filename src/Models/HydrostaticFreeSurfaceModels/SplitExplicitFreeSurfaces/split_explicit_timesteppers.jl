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
##### A low-storage stage needs three buffers per variable -- the substep-start state, the previous stage and
##### the stage being written -- and the substep-start state is the live field until the last stage overwrites
##### it, so the scheme only carries the outputs of the first two stages. The buffers rotate rather than being
##### copied: stage 1 writes slot 1, stage 2 writes slot 2 and stage 3 writes the live fields. The scratch is
##### rebuilt at every substep, so `prognostic_state` returns `nothing`.

"""
    struct RungeKutta3Scheme

Low-storage three-stage Runge-Kutta (1/3, 1/2, 1) substep integrator for the split-explicit barotropic mode.
Within each substep the fast oscillator is advanced from the substep-start state (ηⁿ, Uⁿ, Vⁿ):
```math
U^{(m)} = U^n + γ_m (- g H ∂ η^{(m-1)} + G), \\qquad η^{(m)} = η^n - γ_m ∇ ⋅ U^{(m-1)},
```
with ``γ = (Δτ/3, Δτ/2, Δτ)`` and both stage right-hand sides evaluated at the previous stage.
"""
struct RungeKutta3Scheme{H, U, V}
    η¹ :: H
    U¹ :: U
    V¹ :: V
    η² :: H
    U² :: U
    V² :: V
end

RungeKutta3Scheme() = RungeKutta3Scheme(nothing, nothing, nothing, nothing, nothing, nothing)

# Cells of halo eroded per barotropic substep when halos are not filled in between: one stencil width per
# kernel that advances the state from the previous one.
@inline stages_per_substep(::ForwardBackwardScheme) = 1
@inline stages_per_substep(::RungeKutta3Scheme)     = 3

function materialize_timestepper(::RungeKutta3Scheme, grid, free_surface, velocities, bcs)
    η_bcs = get(bcs, :η, nothing)
    η¹ = free_surface_displacement_field(velocities, free_surface, grid; boundary_conditions = η_bcs)
    η² = free_surface_displacement_field(velocities, free_surface, grid; boundary_conditions = η_bcs)
    U¹ = Field{Face, Center, Nothing}(grid, boundary_conditions = bcs.U)
    V¹ = Field{Center, Face, Nothing}(grid, boundary_conditions = bcs.V)
    U² = Field{Face, Center, Nothing}(grid, boundary_conditions = bcs.U)
    V² = Field{Center, Face, Nothing}(grid, boundary_conditions = bcs.V)
    return RungeKutta3Scheme(η¹, U¹, V¹, η², U², V²)
end

# Per-stage substep fractions γ of Δτ. Each RK stage advances (η, U) from the substep-start state (ηⁿ, Uⁿ), with
# the tendency evaluated at the previous stage: the free surface uses η = ηⁿ + γ(F − ∇·Uᵖ), and the velocity the
# previous-stage thickness ηᵖ, U = Uⁿ + γ(−gH(ηᵖ)∇ηᵖ + G). Genuine (midpoint/low-storage) RK.
@inline stage_parameters(::RungeKutta3Scheme, Δτ::FT) where FT = (Δτ / 3, Δτ / 2, Δτ)

#####
##### Timestepper extrapolations and utils
#####

# A substep integrator may be named rather than instantiated, as `timestepper = :ForwardBackward`.
@inline barotropic_timestepper(name::Symbol) = getglobal(@__MODULE__, Symbol(name, :Scheme))()

materialize_timestepper(name::Symbol, args...) = materialize_timestepper(barotropic_timestepper(name), args...)

@inline stages_per_substep(name::Symbol) = stages_per_substep(barotropic_timestepper(name))

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
    RungeKutta3Scheme(Adapt.adapt(to, ts.η¹),
                      Adapt.adapt(to, ts.U¹),
                      Adapt.adapt(to, ts.V¹),
                      Adapt.adapt(to, ts.η²),
                      Adapt.adapt(to, ts.U²),
                      Adapt.adapt(to, ts.V²))

#####
##### Checkpointing
#####

prognostic_state(::ForwardBackwardScheme) = nothing
restore_prognostic_state!(restored::ForwardBackwardScheme, ::Nothing) = restored

# The Runge-Kutta scratch is recomputed at every substep, so it is not part of the prognostic state.
prognostic_state(::RungeKutta3Scheme) = nothing
restore_prognostic_state!(restored::RungeKutta3Scheme, ::Nothing) = restored
