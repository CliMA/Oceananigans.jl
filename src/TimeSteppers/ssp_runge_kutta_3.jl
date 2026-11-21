"""
    SSPRungeKutta3TimeStepper{C, TG, PF, TI, B} <: AbstractTimeStepper

Hold the coefficients, tendencies and cached states of the three-stage strong-stability-preserving Runge-Kutta
scheme. With a split-explicit free surface, every stage restarts the barotropic solve from `(ηⁿ, Uⁿ)` over the full
`Δt` and blends the increment it returns.

Fields
======
- `Nstages`: number of stages
- `coefficients`: Shu-Osher pairs `(a, b)` of each stage, `Ψᵐ = a Ψⁿ + b (Ψᵐ⁻¹ + Δt Gᵐ)`
- `Gⁿ`: tendency fields at the current stage
- `Ψ⁻`: prognostic fields cached at the beginning of the time step
- `implicit_solver`: solver for implicit vertical diffusion, or `nothing`
- `Ψᵐ⁻¹`: barotropic fields `(η, U, V)` of a split-explicit free surface at the previous stage
"""
struct SSPRungeKutta3TimeStepper{C, TG, PF, TI, B} <: AbstractTimeStepper
    Nstages         :: Int
    coefficients    :: C
    Gⁿ              :: TG
    Ψ⁻              :: PF
    implicit_solver :: TI
    Ψᵐ⁻¹            :: B
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
                                   implicit_solver = nothing,
                                   Gⁿ = map(similar, prognostic_fields),
                                   Ψ⁻ = map(similar, prognostic_fields),
                                   kwargs...)

    coefficients = ((0//1, 1//1), (3//4, 1//4), (1//3, 2//3))
    Ψᵐ⁻¹ = map(similar, haskey(Ψ⁻, :U) ? (; Ψ⁻.η, Ψ⁻.U, Ψ⁻.V) : NamedTuple())

    return SSPRungeKutta3TimeStepper(length(coefficients), coefficients, Gⁿ, Ψ⁻, implicit_solver, Ψᵐ⁻¹)
end

"""
    ssp_quadrature_weights(coefficients)

Return the weights `βₘ` of the stage tendencies in `Ψⁿ⁺¹ = Ψⁿ + Δt Σₘ βₘ Gᵐ`, that is the product of the `b`
coefficients of stage `m` and of every later stage: `(1/6, 1/6, 2/3)` for the three-stage scheme.
"""
ssp_quadrature_weights(coefficients) = ntuple(m -> prod(last, coefficients[m:end]), length(coefficients))

"""
    ssp_stage_fractions(coefficients)

Return the fractions `τᵐ` of `Δt` reached by the state after each stage of the Shu-Osher recursion,
`τᵐ = bᵐ (τᵐ⁻¹ + 1)` with `τ⁰ = 0`: `(1, 1/2, 1)` for the three-stage scheme.
"""
ssp_stage_fractions(coefficients) = accumulate((τ, (a, b)) -> b * (τ + 1), coefficients; init = 0)

Oceananigans.prognostic_state(::SSPRungeKutta3TimeStepper) = nothing

Base.summary(::SSPRungeKutta3TimeStepper) = "SSPRungeKutta3TimeStepper"

function Base.show(io::IO, ts::SSPRungeKutta3TimeStepper)
    print(io, summary(ts), "\n")
    print(io, "├── coefficients: ", ts.coefficients, "\n")
    print(io, "└── implicit_solver: ", summary(ts.implicit_solver))
end

"""
$(TYPEDSIGNATURES)

Step forward `model` one time step `Δt` with the strong-stability-preserving Runge-Kutta scheme: every stage
advances the previous stage by `Δt` and blends the result with the state cached at `tⁿ`. Lagrangian particles
are advanced at the beginning of each stage with the same blend, `Xᵐ = a Xⁿ + b (Xᵐ⁻¹ + Δt Uᵐ⁻¹)`, where `Uᵐ⁻¹`
is the particle velocity at the beginning of the stage (see `substep_lagrangian_particles!`).
"""
function time_step!(model::AbstractModel{<:SSPRungeKutta3TimeStepper}, Δt; callbacks=[])

    kernel_Δt = kernel_time_step(architecture(model.grid), model.grid, Δt)

    maybe_prepare_first_time_step!(model, Δt, callbacks)

    cache_current_fields!(model)
    cache_boundary_conditions!(model)

    stage_times = map(τ -> next_time(model.clock, τ * Δt), ssp_stage_fractions(model.timestepper.coefficients))

    for (stage, (a, b)) in enumerate(model.timestepper.coefficients)

        model.clock.stage = stage
        model.clock.last_stage_Δt = Δt

        # Advance particles with the velocities at the beginning of the stage
        substep_lagrangian_particles!(model, kernel_Δt, stage)

        ssp_substep!(model, kernel_Δt, a, b, callbacks)
        step_closure_prognostics!(model, kernel_Δt)

        model.clock.time = stage_times[stage]

        if stage == model.timestepper.Nstages
            model.clock.last_Δt = Δt
        end

        update_state!(model, callbacks)
    end

    model.clock.iteration += 1

    # Once per time step: the particle state is not reset to tⁿ between stages, so per-stage updates would compound
    update_lagrangian_particle_state!(model)

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
        initialize_boundary_conditions!(model)
        update_state!(model, callbacks)
    end
    return nothing
end

# (σc)ᵐ = a (σc)ⁿ + b (σĉ), with (σc)ⁿ cached in σc⁻
@kernel function _ssp_blend_tracer_field!(c, grid, σc⁻, a, b)
    i, j, k = @index(Global, NTuple)
    σ⁻¹ᶜᶜⁿ = σ⁻¹ⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = a * σc⁻[i, j, k] * σ⁻¹ᶜᶜⁿ + b * c[i, j, k]
end

const MultiStageTimeStepper = Union{SplitRungeKuttaTimeStepper, SSPRungeKutta3TimeStepper}
