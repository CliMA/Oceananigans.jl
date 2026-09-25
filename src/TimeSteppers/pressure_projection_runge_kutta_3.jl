import Oceananigans: prognostic_state, restore_prognostic_state!
using Oceananigans.Fields: CenterField
using Oceananigans.Utils: time_difference_seconds

"""
    PressureProjectionRungeKutta3TimeStepper{FT, TG, TP, TD, TI} <: AbstractTimeStepper

Hold parameters and tendency fields for a low-storage, third-order Runge–Kutta–Wray
time-stepping scheme that solves the pressure Poisson equation **only at the third
substage**. Substages 1 and 2 use a fast-projection (FPJ-α/β) predictor built
from the two most recently stored pseudo-pressures `φⁿ`, `φⁿ⁻¹`, so no Poisson
solve is required at those stages. The Wray coefficients are the same as
[`RungeKutta3TimeStepper`](@ref).

Three named members of the FPJ-α/β family are supported via `(α, β)`:

- `α = 0,    β = 0`    — constant (frozen-pressure) predictor (FPJ-0); 2nd-order in velocity.
- `α = 1,    β = 0`    — linear extrapolation (FPJ-1); 2nd-order in velocity.
- `α = 1//2, β = 1//2` — midpoint-aligned linear interpolation (FPJ-2); **3rd-order in velocity**.

The midpoint variant recovers the same temporal order as standard RK3 with three
Poisson solves while performing only a single Poisson solve per timestep — a
substantial speedup for wall-bounded simulations where the pressure solve
dominates runtime. The constant and linear variants are cheaper to reason about
but cap the velocity order at 2.

The scheme assumes **homogeneous Neumann pressure boundary conditions** on
closed boundaries. Open boundaries and free surfaces are not supported.

References
==========
Le, H. and Moin, P. (1991). An improvement of fractional step methods for the
    incompressible Navier–Stokes equations. Journal of Computational Physics,
    92, 369–379.
Capuano, F., Coppola, G., Chiatto, M. and de Luca, L. (2016). Approximate
    projection method for the incompressible Navier–Stokes equations.
    AIAA Journal, 54, 2178–2181.
De Michele, C., Capuano, F. and Coppola, G. (2020). Fast-projection methods
    for the incompressible Navier–Stokes equations. Fluids, 5, 222.
Aithal, A. B. and Ferrante, A. (2020). A fast pressure-correction method
    for incompressible flows over curved walls. Journal of Computational
    Physics, 421, 109693.
"""
struct PressureProjectionRungeKutta3TimeStepper{FT, TG, TP, TD, TI} <: AbstractTimeStepper
                  γ¹ :: FT
                  γ² :: FT
                  γ³ :: FT
                  ζ² :: FT
                  ζ³ :: FT
                   α :: FT
                   β :: FT
                  Gⁿ :: TG
                  G⁻ :: TG
                  pⁿ :: TP
                pⁿ⁻¹ :: TP
                 Δt⁻¹ :: Base.RefValue{FT}   # Δt of the just-completed step (the one that produced φⁿ)
                 Δt⁻² :: Base.RefValue{FT}   # Δt of the step before that (the one that produced φⁿ⁻¹)
    divergence_buffer :: TD
     implicit_solver :: TI
end

"""
    PressureProjectionRungeKutta3TimeStepper(grid, prognostic_fields;
                                             implicit_solver = nothing,
                                             Gⁿ = map(similar, prognostic_fields),
                                             G⁻ = map(similar, prognostic_fields),
                                             α = 1//2,
                                             β = 1//2)

Return a [`PressureProjectionRungeKutta3TimeStepper`](@ref) on `grid` with the
given `prognostic_fields`. The FPJ-α/β predictor parameters default to the
midpoint choice `α = 1//2, β = 1//2`, which gives 3rd-order accuracy in
velocity. Set `α = β = 0` for the constant (frozen) predictor (FPJ-0) or
`α = 1, β = 0` for the linear-extrapolation predictor (FPJ-1).
"""
function PressureProjectionRungeKutta3TimeStepper(grid, prognostic_fields;
                                                  implicit_solver::TI = nothing,
                                                  Gⁿ::TG = map(similar, prognostic_fields),
                                                  G⁻     = map(similar, prognostic_fields),
                                                  pⁿ::TP = CenterField(grid),
                                                  pⁿ⁻¹   = CenterField(grid),
                                                  divergence_buffer::TD = CenterField(grid),
                                                  α = 1//2,
                                                  β = 1//2) where {TI, TG, TP, TD}

    γ¹ = 8 // 15
    γ² = 5 // 12
    γ³ = 3 // 4

    ζ² = -17 // 60
    ζ³ = -5 // 12

    FT = eltype(grid)
    Δt⁻¹ = Ref(zero(FT))
    Δt⁻² = Ref(zero(FT))

    return PressureProjectionRungeKutta3TimeStepper{FT, TG, TP, TD, TI}(γ¹, γ², γ³, ζ², ζ³, α, β, Gⁿ, G⁻, pⁿ, pⁿ⁻¹, Δt⁻¹, Δt⁻², divergence_buffer, implicit_solver)
end

#####
##### Time stepping
#####

"""
    time_step!(model::AbstractModel{<:PressureProjectionRungeKutta3TimeStepper}, Δt; callbacks=[])

Step forward `model` one timestep `Δt` with the pressure-projection RK3 method.
Only the third substage performs a Poisson solve; substages 1 and 2 apply an
FPJ-α/β predictor projection built from the stored pseudo-pressures `φⁿ`,
`φⁿ⁻¹`.

The first call (`model.clock.iteration == 0`) runs one step of vanilla RK3
(three real Poisson solves) to harvest an honest `φ¹`, then seeds the stored
pressures so the FPJ predictor is well defined from the second step onward.
"""
function time_step!(model, timestepper::PressureProjectionRungeKutta3TimeStepper, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and prepare at iteration 0, in case run! is not used:
    maybe_prepare_first_time_step!(model, Δt, callbacks)

    # The FPJ predictor needs both φⁿ and φⁿ⁻¹ in the timestepper. Run
    # iteration 0 with vanilla RK3 (three real Poisson solves) to harvest an
    # honest φ¹, then transition to the FPJ predictor from iteration 1 onward.
    if model.clock.iteration == 0
        _first_step_vanilla_rk3!(model, timestepper, Δt, callbacks)
        return nothing
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
    # First stage — FPJ-predictor projection (no Poisson solve)
    #

    pressure_projection_rk3_substep!(model, Δt, γ¹, nothing, callbacks, Val(1))
    cache_previous_tendencies!(model)

    tick_stage!(model.clock, first_stage_Δt)

    step_closure_prognostics!(model, first_stage_Δt)
    update_state!(model, callbacks)
    step_lagrangian_particles!(model, first_stage_Δt)

    #
    # Second stage — FPJ-predictor projection (no Poisson solve)
    #

    pressure_projection_rk3_substep!(model, Δt, γ², ζ², callbacks, Val(2))
    cache_previous_tendencies!(model)

    tick_stage!(model.clock, second_stage_Δt)

    step_closure_prognostics!(model, second_stage_Δt)
    update_state!(model, callbacks)
    step_lagrangian_particles!(model, second_stage_Δt)

    #
    # Third stage — full pressure correction (Poisson solve)
    #

    pressure_projection_rk3_substep!(model, Δt, γ³, ζ³, callbacks, Val(3))
    cache_previous_tendencies!(model)

    corrected_third_stage_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_third_stage_Δt, Δt)

    step_closure_prognostics!(model, third_stage_Δt)
    update_state!(model, callbacks)
    step_lagrangian_particles!(model, third_stage_Δt)

    return nothing
end

time_step!(model::AbstractModel{<:PressureProjectionRungeKutta3TimeStepper}, Δt; callbacks=[]) =
    time_step!(model, model.timestepper, Δt; callbacks=callbacks)

# Make sure the clock knows about the first stage Δt
function maybe_prepare_first_time_step!(model::AbstractModel{<:PressureProjectionRungeKutta3TimeStepper}, Δt, callbacks)
    if model.clock.iteration == 0
        γ¹ = model.timestepper.γ¹
        model.clock.last_Δt = Δt
        model.clock.last_stage_Δt = stage_Δt(Δt, γ¹, nothing)
        reconcile_state!(model)
        update_state!(model, callbacks)
    end
    return nothing
end

"""
    pressure_projection_rk3_substep!(model, Δt, γⁿ, ζⁿ, callbacks, ::Val{stage})

Perform a single substep of the pressure-projection RK3 scheme at substage
`stage` (1, 2, or 3). Substages 1 and 2 advance the prognostic fields and
apply an FPJ-α/β predictor projection using the stored `pⁿ`, `pⁿ⁻¹`; substage
3 performs the full pressure correction with a Poisson solve and updates the
stored pseudo-pressures. Must be specialized by each supported model type.
"""
pressure_projection_rk3_substep!(model::AbstractModel, Δt, γ, ζ, callbacks, ::Val) =
    error("pressure_projection_rk3_substep! not implemented for $(typeof(model))")

"""
    seed_pressure_projection_pressures!(model)

Copy `model.pressures.pNHS` (which holds the freshly solved φ¹ at the end of
the vanilla RK3 first step) into both `pⁿ` and `pⁿ⁻¹` of the timestepper. The
second step then has `δφ = pⁿ − pⁿ⁻¹ = 0` and the FPJ predictor degrades to
the constant (FPJ-0) form for that one step regardless of the chosen `(α, β)`;
all subsequent steps use the full FPJ-α/β behavior.
"""
seed_pressure_projection_pressures!(model::AbstractModel) =
    error("seed_pressure_projection_pressures! not implemented for $(typeof(model))")

#####
##### Vanilla RK3 first step
#####
##### The FPJ-α/β substages need both φⁿ and φⁿ⁻¹. We obtain an honest φ¹ by
##### running iteration 0 with the standard RK3 substep (a real Poisson solve
##### at every stage), reusing the timestepper's existing Gⁿ/G⁻ tendency
##### storage — no extra allocations.

function _first_step_vanilla_rk3!(model, timestepper::PressureProjectionRungeKutta3TimeStepper, Δt, callbacks)
    γ¹ = timestepper.γ¹
    γ² = timestepper.γ²
    γ³ = timestepper.γ³
    ζ² = timestepper.ζ²
    ζ³ = timestepper.ζ³

    first_stage_Δt  = stage_Δt(Δt, γ¹, nothing)
    second_stage_Δt = stage_Δt(Δt, γ², ζ²)
    third_stage_Δt  = stage_Δt(Δt, γ³, ζ³)
    tⁿ⁺¹ = next_time(model.clock, Δt)

    rk3_substep!(model, Δt, γ¹, nothing, callbacks)
    cache_previous_tendencies!(model)
    tick_stage!(model.clock, first_stage_Δt)
    step_closure_prognostics!(model, first_stage_Δt)
    update_state!(model, callbacks)
    step_lagrangian_particles!(model, first_stage_Δt)

    rk3_substep!(model, Δt, γ², ζ², callbacks)
    cache_previous_tendencies!(model)
    tick_stage!(model.clock, second_stage_Δt)
    step_closure_prognostics!(model, second_stage_Δt)
    update_state!(model, callbacks)
    step_lagrangian_particles!(model, second_stage_Δt)

    rk3_substep!(model, Δt, γ³, ζ³, callbacks)
    cache_previous_tendencies!(model)
    corrected_third_stage_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
    tick_stage!(model.clock, corrected_third_stage_Δt, Δt)
    step_closure_prognostics!(model, third_stage_Δt)
    update_state!(model, callbacks)
    step_lagrangian_particles!(model, third_stage_Δt)

    seed_pressure_projection_pressures!(model)
    timestepper.Δt⁻¹[] = convert(eltype(timestepper.Δt⁻¹), Δt)
    # Δt⁻² unset on first step. The step-2 predictor uses pⁿ = pⁿ⁻¹ from
    # seeding, so δφ = 0 and the variable-Δt formula's denominator never
    # multiplies anything that survives. By step 3, Δt⁻² will hold Δt from
    # step 1 and the formula is well-defined.
    timestepper.Δt⁻²[] = convert(eltype(timestepper.Δt⁻²), Δt)

    return nothing
end

#####
##### Show methods
#####

function Base.summary(ts::PressureProjectionRungeKutta3TimeStepper{FT}) where FT
    return string("PressureProjectionRungeKutta3TimeStepper{$FT}")
end

function Base.show(io::IO, ts::PressureProjectionRungeKutta3TimeStepper{FT}) where FT
    print(io, "PressureProjectionRungeKutta3TimeStepper{$FT}", '\n')
    print(io, "├── γ: (", ts.γ¹, ", ", ts.γ², ", ", ts.γ³, ")", '\n')
    print(io, "├── ζ: (", ts.ζ², ", ", ts.ζ³, ")", '\n')
    print(io, "├── FPJ parameters: α=", ts.α, ", β=", ts.β, '\n')
    print(io, "├── divergence_buffer: ", summary(ts.divergence_buffer), '\n')
    print(io, "└── implicit_solver: ", isnothing(ts.implicit_solver) ? "nothing" : nameof(typeof(ts.implicit_solver)))
end
