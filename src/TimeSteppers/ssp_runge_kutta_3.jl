"""
    SSPRungeKutta3TimeStepper{C, TG, PF, TI, B, S} <: AbstractTimeStepper

Hold the coefficients, tendencies and cached states of the three-stage strong-stability-preserving Runge-Kutta
scheme, coupled to the barotropic mode as in [Lan et al. (2022)](@cite Lan2022).

Fields
======
- `Nstages`: number of stages
- `coefficients`: Shu-Osher pairs `(a, b)` of each stage, `Ψᵐ = a Ψⁿ + b (Ψᵐ⁻¹ + Δt Gᵐ)`
- `Gⁿ`: tendency fields at the current stage
- `Ψ⁻`: prognostic fields cached at the beginning of the time step
- `implicit_solver`: solver for implicit vertical diffusion, or `nothing`
- `Ψᵐ⁻¹`: free-surface fields at the previous stage
- `Ĝ`: stage-weighted barotropic slow forcing `Σₘ βₘ Ĝᵐ` of a split-explicit free surface
"""
struct SSPRungeKutta3TimeStepper{C, TG, PF, TI, B, S} <: AbstractTimeStepper
    Nstages :: Int
    coefficients :: C
    Gⁿ :: TG
    Ψ⁻ :: PF
    implicit_solver :: TI
    Ψᵐ⁻¹ :: B
    Ĝ :: S
end

"""
    SSPRungeKutta3TimeStepper(grid, prognostic_fields;
                             implicit_solver = nothing,
                             Gⁿ = map(similar, prognostic_fields),
                             Ψ⁻ = map(similar, prognostic_fields))

Return the three-stage strong-stability-preserving Runge-Kutta time stepper in its Shu-Osher form,

    Ψ¹   = Ψⁿ + Δt G(Ψⁿ)
    Ψ²   = 3/4 Ψⁿ + 1/4 (Ψ¹ + Δt G(Ψ¹))
    Ψⁿ⁺¹ = 1/3 Ψⁿ + 2/3 (Ψ² + Δt G(Ψ²))
"""
function SSPRungeKutta3TimeStepper(grid, prognostic_fields, args...;
                                  implicit_solver::TI = nothing,
                                  Gⁿ::TG = map(similar, prognostic_fields),
                                  Ψ⁻::PF = map(similar, prognostic_fields),
                                  kwargs...) where {TI, TG, PF}

    coefficients = ((0//1, 1//1), (3//4, 1//4), (1//3, 2//3))
    Ψᵐ⁻¹ = map(similar, haskey(Ψ⁻, :U) ? (; Ψ⁻.η, Ψ⁻.U, Ψ⁻.V) : haskey(Ψ⁻, :η) ? (; Ψ⁻.η) : NamedTuple())
    Ĝ = map(similar, haskey(Gⁿ, :U) ? (; Gⁿ.U, Gⁿ.V) : NamedTuple())

    return SSPRungeKutta3TimeStepper(length(coefficients), coefficients, Gⁿ, Ψ⁻, implicit_solver, Ψᵐ⁻¹, Ĝ)
end

"""
    ssp_quadrature_weights(coefficients)

Return the weights `βₘ` of the stage tendencies in `Ψⁿ⁺¹ = Ψⁿ + Δt Σₘ βₘ Gᵐ`, that is the product of the `b`
coefficients of stage `m` and of every later stage: `(1/6, 1/6, 2/3)` for the three-stage scheme.
"""
ssp_quadrature_weights(coefficients) = ntuple(m -> prod(coefficients[j][2] for j in m:length(coefficients)), length(coefficients))

prognostic_state(::SSPRungeKutta3TimeStepper) = nothing

Base.summary(::SSPRungeKutta3TimeStepper) = "SSPRungeKutta3TimeStepper"

function Base.show(io::IO, ts::SSPRungeKutta3TimeStepper)
    print(io, summary(ts), "\n")
    print(io, "├── coefficients: ", ts.coefficients, "\n")
    print(io, "└── implicit_solver: ", summary(ts.implicit_solver))
end

"""
$(TYPEDSIGNATURES)

Step forward `model` one time step `Δt` with the strong-stability-preserving Runge-Kutta scheme: every stage
advances the previous stage by `Δt` and blends the result with the state cached at `tⁿ`.
"""
function time_step!(model::AbstractModel{<:SSPRungeKutta3TimeStepper}, Δt; callbacks=[])

    maybe_prepare_first_time_step!(model, Δt, callbacks)

    cache_current_fields!(model)

    # The state after stage m approximates tⁿ + τᵐ Δt, with τᵐ = bᵐ (τᵐ⁻¹ + 1) and τ⁰ = 0.
    stage_fractions = accumulate((τ, (a, b)) -> b * (τ + 1), model.timestepper.coefficients; init = 0)
    stage_times = map(τ -> next_time(model.clock, τ * Δt), stage_fractions)

    for (stage, (a, b)) in enumerate(model.timestepper.coefficients)

        model.clock.stage = stage
        model.clock.last_stage_Δt = Δt

        ssp_substep!(model, Δt, a, b, callbacks)
        step_closure_prognostics!(model, Δt)

        model.clock.time = stage_times[stage]

        if stage == model.timestepper.Nstages
            model.clock.last_Δt = Δt
        end

        update_state!(model, callbacks)
    end

    step_lagrangian_particles!(model, Δt)

    model.clock.iteration += 1

    return nothing
end

"""
    ssp_substep!(model, Δt, a, b, callbacks)

Advance `model` by one strong-stability-preserving stage: a forward-Euler step of `Δt` from the previous stage,
blended with the state cached at `tⁿ` by the Shu-Osher pair `(a, b)`. Implemented by each model type.
"""
function ssp_substep! end

function maybe_prepare_first_time_step!(model::AbstractModel{<:SSPRungeKutta3TimeStepper}, Δt, callbacks)
    if model.clock.iteration == 0
        model.clock.last_Δt = Δt
        model.clock.last_stage_Δt = Δt
        reconcile_state!(model)
        update_state!(model, callbacks)
    end
    return nothing
end

@kernel function _ssp_euler_substep_field!(field, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = field[i, j, k] + Δt * Gⁿ[i, j, k]
end

@kernel function _ssp_blend_field!(field, Ψ⁻, a, b)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = a * Ψ⁻[i, j, k] + b * field[i, j, k]
end

const MultiStageTimeStepper = Union{SplitRungeKuttaTimeStepper, SSPRungeKutta3TimeStepper}
