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
    G★ :: GS  # an η scratch, plus the (U, V) corrector forcing when split-explicit, or `nothing`
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
Auxiliary storage the Shu-Osher blend needs beyond `Gⁿ` and `Ψ⁻`: a scratch copy `η` of the previous
stage's free surface, since the solve restarts from `ηⁿ` whereas the blend applies its increment to
`ηᵐ⁻¹`, plus, for the split-explicit arrangement, the `(U, V)` accumulator for the stage-weighted slow
forcing that drives the corrector sub-cycle. Returns `nothing` for models with no free surface.
"""
@inline function ssp_auxiliary_state(Gⁿ::NamedTuple, Ψ⁻)
    :η in keys(Ψ⁻) || return nothing
    η = similar(Ψ⁻.η)
    has_barotropic = (:U in keys(Gⁿ)) && (:V in keys(Gⁿ))
    return has_barotropic ? (U = similar(Gⁿ.U), V = similar(Gⁿ.V), η = η) : (; η)
end

@inline ssp_auxiliary_state(Gⁿ, Ψ⁻) = nothing

const SSPBarotropicForcing = NamedTuple{(:U, :V, :η)}

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

!!! warning "The barotropic mode is advanced once per step"
    The Shu-Osher weights assume every stage is a forward-Euler step, whereas a sub-cycled barotropic
    solve is a near-exact advance over the full `Δt`. The barotropic velocity is therefore not carried
    through the stages: it is set by a single corrector sub-cycle after them.
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

    maybe_prepare_first_time_step!(model, Δt, callbacks)

    cache_current_fields!(model)

    for (stage, (a, b)) in enumerate(model.timestepper.coefficients)

        model.clock.stage = stage
        model.clock.last_stage_Δt = Δt

        ssp_substep!(model, Δt, a, b, callbacks)
        step_closure_prognostics!(model, Δt)

        if stage == model.timestepper.Nstages
            tick_time!(model.clock, Δt)
        end

        update_state!(model, callbacks)
    end

    step_lagrangian_particles!(model, Δt)

    model.clock.iteration += 1

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
        update_state!(model, callbacks)
    end
    return nothing
end

#####
##### The stage update. With (a, b) = (0, 1) this is exactly forward Euler, which is stage one.
#####

@kernel function _ssp_substep_field!(field, Δt, Gⁿ, Ψ⁻, a, b)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = a * Ψ⁻[i, j, k] + b * (field[i, j, k] + Δt * Gⁿ[i, j, k])
end

#####
##### The corrector's slow forcing: accumulate βₘ Gᵐ across the stages, then hand it to the barotropic
##### solve in place of the final stage's value.
#####
@inline accumulate_ssp_slow_forcing!(ts::SSPRungeKuttaTimeStepper, stage) = accumulate_ssp_slow_forcing!(ts.G★, ts, stage)

@inline accumulate_ssp_slow_forcing!(G★, ts, stage) = nothing

@inline function accumulate_ssp_slow_forcing!(G★::SSPBarotropicForcing, ts, stage)
    β = ssp_quadrature_weights(ts.coefficients)[stage]
    if stage == 1
        parent(G★.U) .= β .* parent(ts.Gⁿ.U)
        parent(G★.V) .= β .* parent(ts.Gⁿ.V)
    else
        parent(G★.U) .+= β .* parent(ts.Gⁿ.U)
        parent(G★.V) .+= β .* parent(ts.Gⁿ.V)
    end
    return nothing
end

@inline install_ssp_slow_forcing!(ts::SSPRungeKuttaTimeStepper) = install_ssp_slow_forcing!(ts.G★, ts)

@inline install_ssp_slow_forcing!(G★, ts) = nothing

@inline function install_ssp_slow_forcing!(G★::SSPBarotropicForcing, ts)
    parent(ts.Gⁿ.U) .= parent(G★.U)
    parent(ts.Gⁿ.V) .= parent(G★.V)
    return nothing
end

"""
    const MultiStageTimeStepper

Timesteppers that advance the state through Runge-Kutta stages, as distinct from the two-level
Adams-Bashforth scheme.
"""
const MultiStageTimeStepper = Union{SplitRungeKuttaTimeStepper, SSPRungeKuttaTimeStepper}
