"""
    SplitRungeKuttaTimeStepper{B, TG, PF, TI} <: AbstractTimeStepper

Hold parameters and tendency fields for a low-storage, n-th order split Runge-Kutta time-stepping scheme.

Fields
======
- `Nstages`: total number of stages
- `β`: Tuple of coefficients for each stage. The number of stages is `length(β)`.
- `Gⁿ`: Tendency fields at the current substep.
- `Ψ⁻`: Prognostic fields cached at the beginning of the time step (before substeps).
- `implicit_solver`: Solver for implicit time stepping of diffusion (or `nothing`).
"""
struct SplitRungeKuttaTimeStepper{B, TG, PF, TI} <: AbstractTimeStepper
    Nstages :: Int
    β  :: B
    Gⁿ :: TG
    Ψ⁻ :: PF # prognostic state at the previous timestep
    implicit_solver :: TI
end

"""
    SplitRungeKuttaTimeStepper(grid, prognostic_fields, args...;
                               implicit_solver::TI = nothing,
                               coefficients = (3, 2, 1),
                               Gⁿ::TG = map(similar, prognostic_fields),
                               Ψ⁻::PF = map(similar, prognostic_fields),
                               kwargs...) where {TI, TG, PF}

Return an ``n``th-order `SplitRungeKuttaTimeStepper` on `grid` and with `tracers`.

The scheme is described by [Wicker and Skamarock (2002)](@cite WickerSkamarock2002); see also
[Silvestri et al. (2026)](@cite Silvestri2026RK3).

In a nutshell, the ``n``th-order low-storage Runge-Kutta timestepper steps forward the state
`Uⁿ` by `Δt` via ``n`` substeps.
A barotropic velocity correction step is applied after at each substep.

The state `U` after each substep `m` is equivalent to an Euler step with a modified time step:

    Δτ   = Δt / βᵐ
    Uᵐ⁺¹ = Uⁿ + Δτ * Gᵐ

where `Uᵐ` is the state at the ``m``-th substep, `Uⁿ` is the state at the ``n``-th timestep,
`Gᵐ` is the tendency at the ``m``-th substep. The coefficients `β` can be specified by the user,
and default to `(3, 2, 1)` for a three-stage scheme. The number of stages is inferred from the
    length of the `β` tuple.

The state at the first substep is taken to be the one that corresponds to the ``n``-th timestep,
`U⁰ = Uⁿ`, and the state after the last substep is then the state at `Uⁿ⁺¹`.

References
==========

* Wicker, Louis J. & Skamarock, William C. (2002). Time-Splitting Methods for Elastic Models
    Using Forward Time Schemes. Monthly Weather Review, 130(8), 2088–2097.
* Silvestri, S., Campin, J.-M., Wagner, G. L., Constantinou, N. C., Lee, X. K., and
    Ferrari, R. (2026). A low-storage Runge-Kutta framework for nonlinear free-surface ocean models.
    J. Adv. Model. Earth Sy. (submitted on Apr 2026; doi:<https://doi.org/10.22541/essoar.15002225/v1>)
"""
function SplitRungeKuttaTimeStepper(grid, prognostic_fields, args...;
                                    implicit_solver::TI = nothing,
                                    coefficients = (3, 2, 1),
                                    Gⁿ::TG = map(similar, prognostic_fields),
                                    Ψ⁻::PF = map(similar, prognostic_fields),
                                    kwargs...) where {TI, TG, PF}

    Nstages = length(coefficients)
    B = typeof(coefficients)
    return SplitRungeKuttaTimeStepper{B, TG, PF, TI}(Nstages, coefficients, Gⁿ, Ψ⁻, implicit_solver)
end

"""
    SplitRungeKuttaTimeStepper(; coefficients=nothing, stages=3)

Construct a `SplitRungeKuttaTimeStepper` by specifying either `coefficients` or number of `stages`.

This simplified constructor creates a "template" time stepper without tendency or state fields,
useful for passing to model constructors which will then build the full time stepper.

Keyword Arguments
=================
- `coefficients`: A tuple of coefficients `(β₁, β₂, ..., βₙ)` for each stage.
- `stages`: Number of stages `n`. If provided, coefficients default to `(n, n-1, ..., 1)`.
            if `coefficients` is specified, this keyword argument is ignored.

Examples
========

Create a 3-stage time stepper with default coefficients (3, 2, 1)

```jldoctest timesteppers
julia> using Oceananigans.TimeSteppers

julia> ts = SplitRungeKuttaTimeStepper(stages=3)
SplitRungeKuttaTimeStepper
├── stages: 3
├── β: (3, 2, 1)
└── implicit_solver: nothing
```

Create a 4-stage time stepper with custom coefficients

```jldoctest timesteppers
julia> ts = SplitRungeKuttaTimeStepper(coefficients=(2, 3, 4, 1))
SplitRungeKuttaTimeStepper
├── stages: 4
├── β: (2, 3, 4, 1)
└── implicit_solver: nothing
```
"""
function SplitRungeKuttaTimeStepper(; coefficients = nothing, stages = 3)
    if isnothing(coefficients) # coefficients takes the priority
        coefficients = tuple(collect(stages:-1:1)...)
    end
    return SplitRungeKuttaTimeStepper{typeof(coefficients), Nothing, Nothing, Nothing}(length(coefficients), coefficients, nothing, nothing, nothing)
end

"""
    spectral_coefficients(c::AbstractVector)

Convert spectral Runge-Kutta coefficients `c` to low-storage coefficients `β` for use
with `SplitRungeKuttaTimeStepper`.

This conversion is useful for designing schemes that minimize dispersion and dissipation
errors; see [Hu et al. (1996)](@cite Hu19996lowdissipation).

# Arguments

- `c`: Vector of spectral coefficients of length `n`.

# Returns

A tuple of low-storage coefficients `(β₁, β₂, ..., βₙ)` where `βᵢ = cₙ₋ᵢ / cₙ₋ᵢ₊₁` for `i < n` and `βₙ = 1`.

# References
* Hu, F. Q., Hussaini, M. Y., & Manthey, J. L. (1996). Low-dissipation and low-dispersion Runge–Kutta
    schemes for computational acoustics. Journal of Computational Physics, 124(1), 177-191.
"""
function spectral_coefficients(c::AbstractVector)
    N = length(c)
    b = similar(c)
    for i in 1:N-1
        b[i] = c[N - i] / c[N - i + 1]
    end
    b[end] = 1
    return tuple(b...)
end

"""
    time_step!(model::AbstractModel{<:SplitRungeKuttaTimeStepper}, Δt; callbacks=[])

Step forward `model` one time step `Δt` using the split Runge-Kutta method.

The split Runge-Kutta scheme advances the model state through `n` substeps, where
`n = model.timestepper.Nstages`. At the beginning of the time step, the current prognostic
fields are cached. Then, for each stage `m`:

1. Compute the `m`-th substep time increment: `Δτ = Δt / βᵐ` (where `β = model.timestepper.β`)
2. Advance the state: `Uᵐ⁺¹ = U⁰ + Δτ * Gᵐ` (where `U⁰` is the cached initial state)
3. Update the `model` state (fill halos, compute diagnostics, etc.)

After all substeps, Lagrangian particles are stepped and the `model.clock`s is advanced.
"""
function time_step!(model::AbstractModel{<:SplitRungeKuttaTimeStepper}, Δt; callbacks=[])

    maybe_prepare_first_time_step!(model, Δt, callbacks)

    cache_current_fields!(model)
    grid = model.grid

    ####
    #### Loop over the stages
    ####

    for (stage, β) in enumerate(model.timestepper.β)

        # Update the clock stage
        Δτ = Δt / β
        model.clock.stage = stage
        model.clock.last_stage_Δt = Δτ

        # Perform the substep
        rk_substep!(model, Δτ, callbacks)

        # Step closure prognostics
        step_closure_prognostics!(model, Δτ)

        # Tick the clock if we ended the stages
        if stage == model.timestepper.Nstages
            tick_time!(model.clock, Δt)
        end

        # Update the state
        update_state!(model, callbacks)
    end

    # Step particles
    step_lagrangian_particles!(model, Δt)

    model.clock.iteration += 1

    return nothing
end

#####
##### These functions need to be implemented by every model independently
#####

"""
    rk_substep!(model::AbstractModel, Δτ, callbacks)

Perform a single Runge-Kutta substep, advancing the model state by `Δτ`.

This is an abstract interface that must be implemented by each model type
(e.g., `NonhydrostaticModel`, `HydrostaticFreeSurfaceModel`, `ShallowWaterModel`).

The implementation should:
1. Compute tendencies for the current state
2. Advance prognostic fields: `U = U⁰ + Δτ * G` (where `U⁰` is the cached initial state)
3. Apply any necessary corrections (e.g., pressure correction for incompressibility)
"""
rk_substep!(model::AbstractModel, Δτ, callbacks) = error("rk_substep! not implemented for $(typeof(model))")

"""
    cache_current_fields!(model::AbstractModel)

Cache the current prognostic fields at the beginning of a split Runge-Kutta time step.

This is an abstract interface that must be implemented by each model type.
The cached fields are stored in `model.timestepper.Ψ⁻` and used as the base state
for all substeps within a single time step.
"""
cache_current_fields!(model::AbstractModel) = error("cache_current_fields! not implemented for $(typeof(model))")

# Make sure the clock knows about the first stage Δt
function maybe_prepare_first_time_step!(model::AbstractModel{<:SplitRungeKuttaTimeStepper}, Δt, callbacks)
    if model.clock.iteration == 0
        model.clock.last_Δt = Δt
        model.clock.last_stage_Δt = Δt / model.timestepper.β[1]
        reconcile_state!(model)
        update_state!(model, callbacks)
    end
    return nothing
end

#####
##### Checkpointing
#####

# SplitRungeKuttaTimeStepper is self-starting!
prognostic_state(ts::SplitRungeKuttaTimeStepper) = nothing
restore_prognostic_state!(restored::SplitRungeKuttaTimeStepper, ::Nothing) = restored

#####
##### Show methods
#####

Base.summary(ts::SplitRungeKuttaTimeStepper) = string("SplitRungeKuttaTimeStepper(", ts.Nstages, ")")

function Base.show(io::IO, ts::SplitRungeKuttaTimeStepper)
    print(io, "SplitRungeKuttaTimeStepper", '\n')
    print(io, "├── stages: ", ts.Nstages, '\n')
    print(io, "├── β: ", ts.β, '\n')
    print(io, "└── implicit_solver: ", isnothing(ts.implicit_solver) ? "nothing" : nameof(typeof(ts.implicit_solver)))
end
