import Oceananigans: prognostic_state, restore_prognostic_state!
using Oceananigans.Fields: CenterField
using Oceananigans.Utils: time_difference_seconds

"""
    LeMoinRungeKutta3TimeStepper{FT, TG, TI} <: AbstractTimeStepper

Hold parameters and tendency fields for a low-storage, third-order Runge–Kutta–Wray
time-stepping scheme that solves the pressure Poisson equation **only at the third
substage**, reusing the most recent pressure (constant-in-time extrapolation) at the
first two substages. The scheme uses the same coefficients as
[`RungeKutta3TimeStepper`](@ref) and follows [Le and Moin (1991)](@cite LeMoin1991).

Compared with the standard RK3, this scheme is second-order accurate in velocity
but performs only one Poisson solve per timestep, which is a substantial speedup for
wall-bounded simulations where the pressure solve dominates runtime.

The scheme assumes **homogeneous Neumann pressure boundary conditions** on closed
boundaries. Open boundaries and free surfaces are not supported.

References
==========
Le, H. and Moin, P. (1991). An improvement of fractional step methods for the incompressible
    Navier–Stokes equations. Journal of Computational Physics, 92, 369–379.
"""
struct LeMoinRungeKutta3TimeStepper{FT, TG, TP, TI} <: AbstractTimeStepper
                 γ¹ :: FT
                 γ² :: FT
                 γ³ :: FT
                 ζ² :: FT
                 ζ³ :: FT
                 Gⁿ :: TG
                 G⁻ :: TG
                 pⁿ :: TP
    implicit_solver :: TI
end

"""
    LeMoinRungeKutta3TimeStepper(grid, prognostic_fields;
                                 implicit_solver = nothing,
                                 Gⁿ = map(similar, prognostic_fields),
                                 G⁻ = map(similar, prognostic_fields))

Return a [`LeMoinRungeKutta3TimeStepper`](@ref) on `grid` with the given
`prognostic_fields`. The cached pressure required by substages 1 and 2 is
*not* stored on the timestepper itself; it is read directly from
`model.pressures.pNHS`, which already holds the actual pressure after substage 3
of the previous timestep (or after the seeding solve performed on the very first
call to `time_step!`).
"""
function LeMoinRungeKutta3TimeStepper(grid, prognostic_fields;
                                      implicit_solver::TI = nothing,
                                      Gⁿ::TG = map(similar, prognostic_fields),
                                      G⁻     = map(similar, prognostic_fields),
                                      pⁿ::TP = CenterField(grid)) where {TI, TG, TP}

    γ¹ = 8 // 15
    γ² = 5 // 12
    γ³ = 3 // 4

    ζ² = -17 // 60
    ζ³ = -5 // 12

    FT = eltype(grid)

    return LeMoinRungeKutta3TimeStepper{FT, TG, TP, TI}(γ¹, γ², γ³, ζ², ζ³, Gⁿ, G⁻, pⁿ, implicit_solver)
end

#####
##### Time stepping
#####

"""
    time_step!(model::AbstractModel{<:LeMoinRungeKutta3TimeStepper}, Δt; callbacks=[])

Step forward `model` one timestep `Δt` with the Le–Moin-style RK3 method. Only the
third substage performs a Poisson solve; substages 1 and 2 reuse the pressure from
the end of the previous timestep.

The first call (`model.clock.iteration == 0`) seeds `model.pressures.pNHS` by
performing one initial Poisson solve on the initial velocity field. This both
populates the cached pressure for the first timestep's substages 1–2 and ensures
the initial velocity is divergence-free, matching the behavior of the standard
[`RungeKutta3TimeStepper`](@ref).
"""
function time_step!(model, timestepper::LeMoinRungeKutta3TimeStepper, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and prepare at iteration 0, in case run! is not used:
    maybe_prepare_first_time_step!(model, callbacks)

    # On the very first call, seed `pNHS` with a Poisson solve on the initial
    # velocities so the cached pressure used by substages 1–2 is well-defined.
    # If the initial velocities are already divergence-free, the seed is zero,
    # which is the correct φ⁰ in that case.
    if model.clock.iteration == 0
        seed_lm_pressure!(model, Δt)
    end

    γ¹ = timestepper.γ¹
    γ² = timestepper.γ²
    γ³ = timestepper.γ³

    ζ¹ = nothing
    ζ² = timestepper.ζ²
    ζ³ = timestepper.ζ³

    first_stage_Δt  = stage_Δt(Δt, γ¹, ζ¹)
    second_stage_Δt = stage_Δt(Δt, γ², ζ²)
    third_stage_Δt  = stage_Δt(Δt, γ³, ζ³)

    tⁿ⁺¹ = next_time(model.clock, Δt)

    #
    # First stage — stale-pressure projection (no Poisson solve)
    #

    lm_rk3_substep!(model, Δt, γ¹, nothing, callbacks, Val(1))
    cache_previous_tendencies!(model)

    tick_stage!(model.clock, first_stage_Δt)

    step_closure_prognostics!(model, first_stage_Δt)
    update_state!(model, callbacks)
    step_lagrangian_particles!(model, first_stage_Δt)

    #
    # Second stage — stale-pressure projection (no Poisson solve)
    #

    lm_rk3_substep!(model, Δt, γ², ζ², callbacks, Val(2))
    cache_previous_tendencies!(model)

    tick_stage!(model.clock, second_stage_Δt)

    step_closure_prognostics!(model, second_stage_Δt)
    update_state!(model, callbacks)
    step_lagrangian_particles!(model, second_stage_Δt)

    #
    # Third stage — full pressure correction (Poisson solve)
    #

    lm_rk3_substep!(model, Δt, γ³, ζ³, callbacks, Val(3))
    cache_previous_tendencies!(model)

    corrected_third_stage_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_third_stage_Δt, Δt)

    step_closure_prognostics!(model, third_stage_Δt)
    update_state!(model, callbacks)
    step_lagrangian_particles!(model, third_stage_Δt)

    return nothing
end

time_step!(model::AbstractModel{<:LeMoinRungeKutta3TimeStepper}, Δt; callbacks=[]) =
    time_step!(model, model.timestepper, Δt; callbacks=callbacks)

"""
    lm_rk3_substep!(model, Δt, γⁿ, ζⁿ, callbacks, ::Val{stage})

Perform a single substep of the Le–Moin-style RK3 scheme at substage `stage` (1, 2, or 3).
Substages 1 and 2 advance the prognostic fields and apply a stale-pressure projection
using `model.pressures.pNHS`; substage 3 performs the full pressure correction with a
Poisson solve. Must be specialized by each supported model type.
"""
lm_rk3_substep!(model::AbstractModel, Δt, γ, ζ, callbacks, ::Val) =
    error("lm_rk3_substep! not implemented for $(typeof(model))")

"""
    seed_lm_pressure!(model, Δt)

Perform a single Poisson solve on the initial velocity field to populate
`model.pressures.pNHS` with a coherent starting pressure φ⁰ for the
Le–Moin substages 1 and 2 to project against.
"""
seed_lm_pressure!(model::AbstractModel, Δt) =
    error("seed_lm_pressure! not implemented for $(typeof(model))")

#####
##### Show methods
#####

function Base.summary(ts::LeMoinRungeKutta3TimeStepper{FT}) where FT
    return string("LeMoinRungeKutta3TimeStepper{$FT}")
end

function Base.show(io::IO, ts::LeMoinRungeKutta3TimeStepper{FT}) where FT
    print(io, "LeMoinRungeKutta3TimeStepper{$FT}", '\n')
    print(io, "├── γ: (", ts.γ¹, ", ", ts.γ², ", ", ts.γ³, ")", '\n')
    print(io, "├── ζ: (", ts.ζ², ", ", ts.ζ³, ")", '\n')
    print(io, "└── implicit_solver: ", isnothing(ts.implicit_solver) ? "nothing" : nameof(typeof(ts.implicit_solver)))
end
