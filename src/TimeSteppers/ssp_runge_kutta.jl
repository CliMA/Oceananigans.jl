"""
    SSPRungeKuttaTimeStepper{C, TG, PF, TI} <: AbstractTimeStepper

Hold parameters and tendency fields for a strong-stability-preserving Runge-Kutta time-stepping scheme
in the split-explicit arrangement of [Lan et al. (2022)](@cite Lan2022).

Fields
======
- `Nstages`: total number of stages
- `coefficients`: Shu-Osher pairs `(a, b)`, one per stage; the stage advance is
  `Ψᵐ = a Ψⁿ + b (Ψᵐ⁻¹ + Δt Gᵐ)`
- `Gⁿ`: Tendency fields at the current stage.
- `Ψ⁻`: Prognostic fields cached at the beginning of the time step (before the stages).
- `implicit_solver`: Solver for implicit time stepping of diffusion (or `nothing`).

Unlike `SplitRungeKuttaTimeStepper`, every stage advances by the *full* `Δt` and the stages are combined
by the Shu-Osher convex weights rather than starting afresh from `Ψⁿ` with a reduced increment.
"""
struct SSPRungeKuttaTimeStepper{C, TG, PF, TI, GS} <: AbstractTimeStepper
    Nstages :: Int
    coefficients :: C
    Gⁿ :: TG
    Ψ⁻ :: PF
    implicit_solver :: TI
    G★ :: GS  # the previous-stage (η, U, V) and the (U, V) corrector forcing when split-explicit, or `nothing`
end

"""
    ssp_quadrature_weights(coefficients)

Return the effective quadrature weights `βₘ` on the stage tendencies, i.e. the coefficients such that
`Ψⁿ⁺¹ = Ψⁿ + Δt Σₘ βₘ Gᵐ`. For the Shu-Osher recursion `Ψᵐ = a Ψⁿ + b (Ψᵐ⁻¹ + Δt Gᵐ)` the weight of stage
`m` is the product of the `b` coefficients of stage `m` and every stage after it, `(1/6, 1/6, 2/3)` for
the three-stage scheme.
"""
@inline function ssp_quadrature_weights(coefficients)
    N = length(coefficients)
    return ntuple(m -> prod(coefficients[j][2] for j in m:N), N)
end

"""
    ssp_stage_fractions(coefficients)

Return the fractions `τᵐ` of `Δt` reached by the state after each stage of the Shu-Osher recursion,
`τᵐ = bᵐ (τᵐ⁻¹ + 1)` with `τ⁰ = 0`: `(1, 1/2, 1)` for the three-stage scheme.
"""
ssp_stage_fractions(coefficients) = accumulate((τ, (a, b)) -> b * (τ + 1), coefficients; init = 0)

"""
Auxiliary storage the split-explicit arrangement needs beyond `Gⁿ` and `Ψ⁻`: scratch copies of the previous
stage's barotropic state, since every solve restarts from `(ηⁿ, Uⁿ, Vⁿ)` whereas the blend applies its increment
to `(ηᵐ⁻¹, Uᵐ⁻¹, Vᵐ⁻¹)`, and the `(U, V)` accumulator for the stage-weighted slow forcing that drives the
corrector sub-cycle. Returns `nothing` for any other free surface.
"""
@inline function ssp_auxiliary_state(Gⁿ::NamedTuple, Ψ⁻)
    (:U in keys(Gⁿ)) && (:V in keys(Gⁿ)) || return nothing
    return (U = similar(Gⁿ.U), V = similar(Gⁿ.V), Uᵐ = similar(Ψ⁻.U), Vᵐ = similar(Ψ⁻.V), η = similar(Ψ⁻.η))
end

@inline ssp_auxiliary_state(Gⁿ, Ψ⁻) = nothing

const SSPBarotropicForcing = NamedTuple{(:U, :V, :Uᵐ, :Vᵐ, :η)}

# Shu-Osher pairs (a, b) for the classical three-stage strong-stability-preserving scheme.
const SSPRK3_COEFFICIENTS = ((0//1, 1//1), (3//4, 1//4), (1//3, 2//3))

"""
    SSPRungeKuttaTimeStepper(grid, prognostic_fields, args...;
                             implicit_solver = nothing,
                             coefficients = SSPRK3_COEFFICIENTS,
                             Gⁿ = map(similar, prognostic_fields),
                             Ψ⁻ = map(similar, prognostic_fields))

Return a strong-stability-preserving `SSPRungeKuttaTimeStepper` on `grid`.

The three-stage scheme is the classical Shu-Osher form,
```math
\\begin{aligned}
Ψ^{(1)} &= Ψ^n + Δt\\,G(Ψ^n) \\\\
Ψ^{(2)} &= \\tfrac34 Ψ^n + \\tfrac14 \\left[Ψ^{(1)} + Δt\\,G(Ψ^{(1)})\\right] \\\\
Ψ^{n+1} &= \\tfrac13 Ψ^n + \\tfrac23 \\left[Ψ^{(2)} + Δt\\,G(Ψ^{(2)})\\right]
\\end{aligned}
```
which for a linear problem has the same amplification polynomial as the low-storage form of
`SplitRungeKuttaTimeStepper`, and therefore the same un-split stability limit. Paired with the barotropic
sub-cycling arrangement of [Lan et al. (2022)](@cite Lan2022) it has no resonance near a barotropic
Courant number of `5.7`, where the low-storage composition loses about a third of its damping.

!!! note "The barotropic mode is blended by increment, not by state"
    The Shu-Osher weights assume every stage is a forward-Euler step, whereas a sub-cycled barotropic
    solve is a near-exact advance over the full `Δt`, so composing three such advances would
    over-integrate the barotropic mode. The blend therefore acts on the *increment* each sub-cycle
    produces, `Ψᵐ = a Ψⁿ + b (Ψᵐ⁻¹ + Ψ⋆ - Ψⁿ)`: every sub-cycle restarts from `Ψⁿ` and the weights sum
    to one, so the free barotropic propagator is applied exactly once per step while the responses to
    the stage forcings are combined as the Shu-Osher recursion requires.
"""
function SSPRungeKuttaTimeStepper(grid, prognostic_fields, args...;
                                  implicit_solver::TI = nothing,
                                  coefficients = SSPRK3_COEFFICIENTS,
                                  Gⁿ::TG = map(similar, prognostic_fields),
                                  Ψ⁻::PF = map(similar, prognostic_fields),
                                  kwargs...) where {TI, TG, PF}

    !isnothing(implicit_solver) &&
        @warn("Implicit-explicit time-stepping with an SSP Runge-Kutta scheme is not tested.")

    G★ = ssp_auxiliary_state(Gⁿ, Ψ⁻)
    C, GS = typeof(coefficients), typeof(G★)
    return SSPRungeKuttaTimeStepper{C, TG, PF, TI, GS}(length(coefficients), coefficients,
                                                       Gⁿ, Ψ⁻, implicit_solver, G★)
end

"""
    SSPRungeKuttaTimeStepper(; coefficients = SSPRK3_COEFFICIENTS)

Construct an `SSPRungeKuttaTimeStepper` from its Shu-Osher `coefficients`, defaulting to the classical
three-stage scheme.

The fields are left empty: the result carries the `coefficients` to a model constructor, which calls
`TimeStepper` on it once the grid and prognostic fields are known.
"""
function SSPRungeKuttaTimeStepper(; coefficients = SSPRK3_COEFFICIENTS)
    C = typeof(coefficients)
    return SSPRungeKuttaTimeStepper{C, Nothing, Nothing, Nothing, Nothing}(length(coefficients), coefficients,
                                                                          nothing, nothing, nothing, nothing)
end

Base.summary(ts::SSPRungeKuttaTimeStepper) = "SSPRungeKuttaTimeStepper($(ts.Nstages) stages)"

function Base.show(io::IO, ts::SSPRungeKuttaTimeStepper)
    print(io, summary(ts), "\n")
    print(io, "├── stages: ", ts.Nstages, "\n")
    print(io, "├── coefficients: ", ts.coefficients, "\n")
    print(io, "└── implicit_solver: ", summary(ts.implicit_solver))
end

"""
$(TYPEDSIGNATURES)

Step forward `model` one time step `Δt` with a strong-stability-preserving Runge-Kutta scheme.

Every stage advances by the full `Δt` and is blended with the cached state `Ψⁿ` by the Shu-Osher
coefficients.
"""
function time_step!(model::AbstractModel{<:SSPRungeKuttaTimeStepper}, Δt; callbacks=[])

    kernel_Δt = kernel_time_step(architecture(model.grid), model.grid, Δt)

    maybe_prepare_first_time_step!(model, Δt, callbacks)

    cache_current_fields!(model)
    cache_boundary_conditions!(model)

    stage_times = map(τ -> next_time(model.clock, τ * Δt), ssp_stage_fractions(model.timestepper.coefficients))

    for (stage, (a, b)) in enumerate(model.timestepper.coefficients)

        model.clock.stage = stage
        model.clock.last_stage_Δt = Δt

        ssp_substep!(model, kernel_Δt, a, b, callbacks)
        step_closure_prognostics!(model, kernel_Δt)

        model.clock.time = stage_times[stage]

        if stage == model.timestepper.Nstages
            model.clock.last_Δt = Δt
        end

        update_state!(model, callbacks)
    end

    step_lagrangian_particles!(model, kernel_Δt)

    model.clock.iteration += 1

    # Once per time step: the particle state is not reset to tⁿ between stages, so per-stage updates would compound
    update_lagrangian_particle_state!(model)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Perform a single strong-stability-preserving Runge-Kutta stage, advancing the model state by `Δt` and
blending it with the cached state by the Shu-Osher pair `(a, b)`.

Implemented by each model type, as `rk_substep!` is.
"""
ssp_substep!(model::AbstractModel, Δt, a, b, callbacks) =
    error("ssp_substep! not implemented for $(typeof(model))")

function maybe_prepare_first_time_step!(model::AbstractModel{<:SSPRungeKuttaTimeStepper}, Δt, callbacks)
    if model.clock.iteration == 0
        model.clock.last_Δt = Δt
        model.clock.last_stage_Δt = Δt
        reconcile_state!(model)
        initialize_boundary_conditions!(model)
        update_state!(model, callbacks)
    end
    return nothing
end

#####
##### The stage update: the Euler step Ψ̂ᵐ = Ψᵐ⁻¹ + Δt Gᵐ, and the Shu-Osher blend Ψᵐ = a Ψⁿ + b Ψ̂ᵐ.
#####

@kernel function _ssp_euler_substep_field!(field, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = field[i, j, k] + Δt * Gⁿ[i, j, k]
end

@kernel function _ssp_blend_field!(field, Ψ⁻, a, b)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = a * Ψ⁻[i, j, k] + b * field[i, j, k]
end

# (σc)ᵐ = a (σc)ⁿ + b (σĉ), with (σc)ⁿ cached in σc⁻
@kernel function _ssp_blend_tracer_field!(c, grid, σc⁻, a, b)
    i, j, k = @index(Global, NTuple)
    σ⁻¹ᶜᶜⁿ = σ⁻¹ⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = a * σc⁻[i, j, k] * σ⁻¹ᶜᶜⁿ + b * c[i, j, k]
end

"""
    const MultiStageTimeStepper

Timesteppers that advance the state through Runge-Kutta stages, as distinct from the two-level
Adams-Bashforth scheme.
"""
const MultiStageTimeStepper = Union{SplitRungeKuttaTimeStepper, SSPRungeKuttaTimeStepper}
