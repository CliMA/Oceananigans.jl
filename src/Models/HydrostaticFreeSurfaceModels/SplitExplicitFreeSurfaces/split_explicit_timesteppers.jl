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
##### `RungeKutta2Scheme` (2-stage midpoint) and `RungeKutta3Scheme` (3-stage low-storage) advance the fast
##### (η, U, V) oscillator with genuine RK stages per barotropic substep. Both carry substep-start scratch
##### (η⁰, U⁰, V⁰) plus a previous-stage free surface (ηᵖ). The scratch is not prognostic (it is recomputed at
##### every substep), so `prognostic_state` returns `nothing`.

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
end

RungeKutta3Scheme() = RungeKutta3Scheme(nothing, nothing, nothing, nothing)

"""
    struct RungeKutta2Scheme

Two-stage midpoint Runge-Kutta substep integrator for the split-explicit barotropic mode: ``γ = (Δτ/2, Δτ)``,
both stage right-hand sides evaluated at the previous stage. Second order and cheaper than RK3 — but plain RK2 is
weakly amplifying on the imaginary axis (the barotropic oscillator), so it relies on the averaging filter's
dissipation for stability. Viable only if order 2 is the target and the filter absorbs the growth.
"""
struct RungeKutta2Scheme{H, U, V, P}
    η⁰ :: H
    U⁰ :: U
    V⁰ :: V
    ηᵖ :: P
end

RungeKutta2Scheme() = RungeKutta2Scheme(nothing, nothing, nothing, nothing)

const MultiStageScheme = Union{RungeKutta2Scheme, RungeKutta3Scheme}

@inline requires_multistage(::ForwardBackwardScheme) = false
@inline requires_multistage(::MultiStageScheme)      = true

function materialize_timestepper(ts::MultiStageScheme, grid, free_surface, velocities, u_bcs, v_bcs)
    η⁰ = free_surface_displacement_field(velocities, free_surface, grid)
    ηᵖ = free_surface_displacement_field(velocities, free_surface, grid)
    U⁰ = Field{Face, Center, Nothing}(grid, boundary_conditions = u_bcs)
    V⁰ = Field{Center, Face, Nothing}(grid, boundary_conditions = v_bcs)
    return Base.typename(typeof(ts)).wrapper(η⁰, U⁰, V⁰, ηᵖ)
end

# Per-stage substep fractions γ of Δτ. Each RK stage advances (η, U) from the substep-start state (η⁰, U⁰), with
# the tendency evaluated at the previous stage: the free surface uses η = η⁰ + γ(F − ∇·U), then the velocity uses
# the previous-stage thickness ηᵖ, U = U⁰ + γ(−gH(ηᵖ)∇ηᵖ + G). Genuine (midpoint/low-storage) RK.
@inline stage_parameters(::RungeKutta2Scheme, Δτ::FT) where FT = (Δτ / 2, Δτ)
@inline stage_parameters(::RungeKutta3Scheme, Δτ::FT) where FT = (Δτ / 3, Δτ / 2, Δτ)

#####
##### Adams-Bashforth 3 substep integrator
#####

"""
    struct AdamsBashforth3Scheme

Generalized forward–backward (Adams–Bashforth-3 / Adams–Moulton-4) substep integrator for the split-explicit
barotropic mode, following Shchepetkin & McWilliams (2005, sec. 2.3) as modified by Demange et al. (2019). It
reuses the plain forward–backward substep path (η-first, like the multistage schemes) through the `η★`/`U★`/`V★`
hooks:
```math
η^{m+1} = η^m - Δτ\\, ∇·U★ + Δτ\\, F , \\qquad
U^{m+1} = U^m - Δτ\\, gH\\, ∇η★ + Δτ\\, G ,
```
with the AB3 **forward extrapolation** of the transport
``U★ = (\\tfrac32+β)U^m - (\\tfrac12+2β)U^{m-1} + β U^{m-2}`` advancing the free surface, and the AM4-type
**forward-weighted** free surface ``η★ = (\\tfrac12+γ+2ε)η^{m+1} + (…)η^m + (γ+3ε)η^{m-1} - ε η^{m-2}`` — which
includes the just-updated ``η^{m+1}`` (the backward feedback) — advancing the velocity. `U★`/`V★` are also the
transport carried into the tracer-continuity average, so constancy is preserved. Both stencils sum to one, so a
constant state and the frozen slow forcing are untouched. Unlike the plain `ForwardBackwardScheme` (1st-order
coupling), this generalized-FB substep is 3rd-order accurate in time, lifting the coupled baroclinic accuracy
toward second order like `RungeKutta2Scheme`/`RungeKutta3Scheme`, at one tendency evaluation per substep. It
carries three free-surface (``ηᵐ⁻¹, ηᵐ⁻², ηᵐ⁻³``) and two velocity (``Uᵐ⁻¹, Uᵐ⁻²`` and the same for ``V``)
history levels, seeded from the sub-cycle-start state; the scratch is reset every sub-cycle, so
`prognostic_state` returns `nothing`.
"""
struct AdamsBashforth3Scheme{H, U, V}
    ηᵐ⁻¹ :: H
    ηᵐ⁻² :: H
    ηᵐ⁻³ :: H
    Uᵐ⁻¹ :: U
    Uᵐ⁻² :: U
    Vᵐ⁻¹ :: V
    Vᵐ⁻² :: V
end

AdamsBashforth3Scheme() = AdamsBashforth3Scheme(nothing, nothing, nothing, nothing, nothing, nothing, nothing)

@inline requires_multistage(::AdamsBashforth3Scheme) = false

# SM05 / Demange generalized-FB coefficients (non-dissipative). `AB3_EXTRAP` is the AB3 forward extrapolation of
# the transport applied in `U★`/`V★` (which advance the free surface); `AB3_WEIGHT` is the AM4 backward
# weighting of the free surface — including the just-updated ηᵐ⁺¹ — applied in `η★` (which advances the
# velocity). Both sum to one, so a constant state and the frozen slow forcing are left untouched.
const AB3_EXTRAP = (1.781105, -1.062210, 0.281105)                       # β = 0.281105
const AB3_WEIGHT = (0.60296872, 0.30382442, 0.08344500, 0.00976186)      # γ = 0.083445, ε = 0.00976186

function materialize_timestepper(::AdamsBashforth3Scheme, grid, free_surface, velocities, u_bcs, v_bcs)
    ηᵐ⁻¹ = free_surface_displacement_field(velocities, free_surface, grid)
    ηᵐ⁻² = free_surface_displacement_field(velocities, free_surface, grid)
    ηᵐ⁻³ = free_surface_displacement_field(velocities, free_surface, grid)
    Uᵐ⁻¹ = Field{Face, Center, Nothing}(grid, boundary_conditions = u_bcs)
    Uᵐ⁻² = Field{Face, Center, Nothing}(grid, boundary_conditions = u_bcs)
    Vᵐ⁻¹ = Field{Center, Face, Nothing}(grid, boundary_conditions = v_bcs)
    Vᵐ⁻² = Field{Center, Face, Nothing}(grid, boundary_conditions = v_bcs)
    return AdamsBashforth3Scheme(ηᵐ⁻¹, ηᵐ⁻², ηᵐ⁻³, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻²)
end

# Seed every history level with the current state at the start of each sub-cycle, so the first substeps degrade
# smoothly to forward-backward before the multistep stencils build up.
function initialize_free_surface_timestepper!(ts::AdamsBashforth3Scheme, η, U, V)
    parent(ts.ηᵐ⁻¹) .= parent(η); parent(ts.ηᵐ⁻²) .= parent(η); parent(ts.ηᵐ⁻³) .= parent(η)
    parent(ts.Uᵐ⁻¹) .= parent(U); parent(ts.Uᵐ⁻²) .= parent(U)
    parent(ts.Vᵐ⁻¹) .= parent(V); parent(ts.Vᵐ⁻²) .= parent(V)
    return nothing
end

Adapt.adapt_structure(to, ts::AdamsBashforth3Scheme) =
    AdamsBashforth3Scheme(Adapt.adapt(to, ts.ηᵐ⁻¹), Adapt.adapt(to, ts.ηᵐ⁻²), Adapt.adapt(to, ts.ηᵐ⁻³),
                          Adapt.adapt(to, ts.Uᵐ⁻¹), Adapt.adapt(to, ts.Uᵐ⁻²),
                          Adapt.adapt(to, ts.Vᵐ⁻¹), Adapt.adapt(to, ts.Vᵐ⁻²))

prognostic_state(::AdamsBashforth3Scheme) = nothing
restore_prognostic_state!(restored::AdamsBashforth3Scheme, ::Nothing) = restored

#####
##### Timestepper extrapolations and utils
#####

function materialize_timestepper(name::Symbol, args...)
    fullname = Symbol(name, :Scheme)
    TS = getglobal(@__MODULE__, fullname)
    return materialize_timestepper(TS, args...)
end

initialize_free_surface_timestepper!(::ForwardBackwardScheme, args...) = nothing
initialize_free_surface_timestepper!(::MultiStageScheme, args...) = nothing

# The functions `η★`, `U★` and `V★` represent the free surface and barotropic velocities as they enter the
# opposite update: `η★` the free surface used in the velocity update, `U★`/`V★` the transport used in the
# free-surface update (and carried into the tracer-continuity average). Plain forward-backward simply returns
# the current field.
@inline U★(i, j, k, grid,  ::ForwardBackwardScheme, Uᵐ)   = @inbounds Uᵐ[i, j, k]
@inline V★(i, j, k, grid,  ::ForwardBackwardScheme, Vᵐ)   = @inbounds Vᵐ[i, j, k]
@inline η★(i, j, k, grid,  ::ForwardBackwardScheme, ηᵐ⁺¹) = @inbounds ηᵐ⁺¹[i, j, k]

@inline cache_previous_free_surface!(::ForwardBackwardScheme, i, j, k, η)    = nothing
@inline   cache_previous_velocities!(::ForwardBackwardScheme, i, j, k, U, V) = nothing

# Generalized forward-backward: AM4 backward weighting of the free surface (including the just-updated ηᵐ⁺¹)
# into the velocity update, AB3 forward extrapolation of the transport into the free-surface update.
@inline η★(i, j, k, grid, ts::AdamsBashforth3Scheme, η) =
    @inbounds AB3_WEIGHT[1] * η[i, j, k] + AB3_WEIGHT[2] * ts.ηᵐ⁻¹[i, j, k] + AB3_WEIGHT[3] * ts.ηᵐ⁻²[i, j, k] + AB3_WEIGHT[4] * ts.ηᵐ⁻³[i, j, k]
@inline U★(i, j, k, grid, ts::AdamsBashforth3Scheme, U) =
    @inbounds AB3_EXTRAP[1] * U[i, j, k] + AB3_EXTRAP[2] * ts.Uᵐ⁻¹[i, j, k] + AB3_EXTRAP[3] * ts.Uᵐ⁻²[i, j, k]
@inline V★(i, j, k, grid, ts::AdamsBashforth3Scheme, V) =
    @inbounds AB3_EXTRAP[1] * V[i, j, k] + AB3_EXTRAP[2] * ts.Vᵐ⁻¹[i, j, k] + AB3_EXTRAP[3] * ts.Vᵐ⁻²[i, j, k]

@inline function cache_previous_free_surface!(ts::AdamsBashforth3Scheme, i, j, k, η)
    @inbounds ts.ηᵐ⁻³[i, j, k] = ts.ηᵐ⁻²[i, j, k]; ts.ηᵐ⁻²[i, j, k] = ts.ηᵐ⁻¹[i, j, k]; ts.ηᵐ⁻¹[i, j, k] = η[i, j, k]
    return nothing
end

@inline function cache_previous_velocities!(ts::AdamsBashforth3Scheme, i, j, k, U, V)
    @inbounds begin
        ts.Uᵐ⁻²[i, j, k] = ts.Uᵐ⁻¹[i, j, k]; ts.Uᵐ⁻¹[i, j, k] = U[i, j, k]
        ts.Vᵐ⁻²[i, j, k] = ts.Vᵐ⁻¹[i, j, k]; ts.Vᵐ⁻¹[i, j, k] = V[i, j, k]
    end
    return nothing
end

#####
##### Adapt
#####

Adapt.adapt_structure(to, ts::MultiStageScheme) =
    Base.typename(typeof(ts)).wrapper(Adapt.adapt(to, ts.η⁰),
                                      Adapt.adapt(to, ts.U⁰),
                                      Adapt.adapt(to, ts.V⁰),
                                      Adapt.adapt(to, ts.ηᵖ))

#####
##### Checkpointing
#####

prognostic_state(::ForwardBackwardScheme) = nothing
restore_prognostic_state!(restored::ForwardBackwardScheme, ::Nothing) = restored

# The multi-stage scratch is recomputed at every substep, so it is not part of the prognostic state.
prognostic_state(::MultiStageScheme) = nothing
restore_prognostic_state!(restored::MultiStageScheme, ::Nothing) = restored
