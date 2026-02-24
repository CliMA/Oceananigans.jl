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

Return a nth-order `SplitRungeKuttaTimeStepper` on `grid` and with `tracers`.
The tendency fields `Gⁿ`, and the previous state `Ψ⁻` can be modified via optional `kwargs`.

The scheme is described by [Wicker and Skamarock (2002)](@cite WickerSkamarock2002). In a nutshell,
the nth-order low-storage Runge-Kutta timestepper steps forward the state `Uⁿ` by `Δt` via n substeps.
A barotropic velocity correction step is applied after at each substep.

The state `U` after each substep `m` is equivalent to an Euler step with a modified time step:

    Δt̃   = Δt / βᵐ
    Uᵐ⁺¹ = Uⁿ + Δt̃ * Gᵐ

where `Uᵐ` is the state at the ``m``-th substep, `Uⁿ` is the state at the ``n``-th timestep,
`Gᵐ` is the tendency at the ``m``-th substep. The coefficients `β` can be specified by the user,
and default to `(3, 2, 1)` for a three-stage scheme. The number of stages is inferred from the length of the
`β` tuple.

The state at the first substep is taken to be the one that corresponds to the ``n``-th timestep,
`U¹ = Uⁿ`, and the state after the last substep is then the state at `Uⁿ⁺¹`.

References
==========

Wicker, Louis J. & Skamarock, William C. (2002). Time-Splitting Methods for Elastic Models
    Using Forward Time Schemes. Monthly Weather Review, 130(8), 2088–2097.
"""
function SplitRungeKuttaTimeStepper(grid, prognostic_fields, args...;
                                    implicit_solver::TI = nothing,
                                    coefficients = (3, 2, 1),
                                    Gⁿ::TG = map(similar, prognostic_fields),
                                    Ψ⁻::PF = map(similar, prognostic_fields),
                                    kwargs...) where {TI, TG, PF}

    return SplitRungeKuttaTimeStepper{typeof(coefficients), TG, PF, TI}(length(coefficients), coefficients, Gⁿ, Ψ⁻, implicit_solver)
end

"""
    SplitRungeKuttaTimeStepper(; coefficients=nothing, stages=3)

Construct a `SplitRungeKuttaTimeStepper` by specifying either `coefficients` or `stages`.

This simplified constructor creates a "template" time stepper without tendency or state fields,
useful for passing to model constructors which will then build the full time stepper.

Keyword Arguments
=================
- `coefficients`: A tuple of coefficients `(β₁, β₂, ..., βₙ)` for each stage.
- `stages`: Number of stages `n`. If provided, coefficients default to `(n, n-1, ..., 1)`.
            if `coefficients` is specified, this keyword argument is ignored.

Example
=======
```julia
# Create a 3-stage time stepper with default coefficients (3, 2, 1)
ts = SplitRungeKuttaTimeStepper(stages=3)

# Create a 4-stage time stepper with custom coefficients
ts = SplitRungeKuttaTimeStepper(coefficients=(4, 3, 2, 1))
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

- `c`: Vector of spectral coefficients of length `N`.

# Returns

A tuple of low-storage coefficients `(β₁, β₂, ..., βₙ)` where `βᵢ = cₙ₋ᵢ / cₙ₋ᵢ₊₁` for `i < N` and `βₙ = 1`.

# References
* Hu, F. Q., Hussaini, M. Y., & Manthey, J. L. (1996). Low-dissipation and low-dispersion Runge–Kutta
    schemes for computational acoustics. Journal of computational physics, 124(1), 177-191.
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

The split Runge-Kutta scheme advances the model state through `n` substeps (where `n = model.timestepper.Nstages`).
At the beginning of the time step, the current prognostic fields are cached. Then, for each stage `m`:

1. Compute the substep time increment: `Δτ = Δt / βᵐ`
2. Advance the state: `Uᵐ⁺¹ = U⁰ + Δτ * Gᵐ` (where `U⁰` is the cached initial state)
3. Update the model state (fill halos, compute diagnostics, etc.)

After all substeps, Lagrangian particles are stepped and the clock is advanced.
"""
function time_step!(model::AbstractModel{<:SplitRungeKuttaTimeStepper}, Δt; callbacks=[])

    maybe_initialize_state!(model, callbacks)

    cache_current_fields!(model)
    grid = model.grid

    ####
    #### Loop over the stages
    ####

    for (stage, β) in enumerate(model.timestepper.β)
        # Update the clock stage
        model.clock.stage = stage

        # Perform the substep
        Δτ = Δt / β
        rk_substep!(model, Δτ, callbacks)

        # Step closure prognostics and update the state
        step_closure_prognostics!(model, Δτ)
        update_state!(model, callbacks)
    end

    # Finalize step
    step_lagrangian_particles!(model, Δt)
    tick!(model.clock, Δt)

    return nothing
end

#####
##### These functions need to be implemented by every model independently
#####

"""
    rk_substep!(model::AbstractModel, Δt, callbacks)

Perform a single Runge-Kutta substep, advancing the model state by `Δτ`.

This is an abstract interface that must be implemented by each model type
(e.g., `NonhydrostaticModel`, `HydrostaticFreeSurfaceModel`, `ShallowWaterModel`).

The implementation should:
1. Compute tendencies for the current state
2. Advance prognostic fields: `U = U⁰ + Δτ * G` (where `U⁰` is the cached initial state)
3. Apply any necessary corrections (e.g., pressure correction for incompressibility)
"""
rk_substep!(model::AbstractModel, Δt, callbacks) = error("rk_substep! not implemented for $(typeof(model))")

"""
    cache_current_fields!(model::AbstractModel)

Cache the current prognostic fields at the beginning of a split Runge-Kutta time step.

This is an abstract interface that must be implemented by each model type.
The cached fields are stored in `model.timestepper.Ψ⁻` and used as the base state
for all substeps within a single time step.
"""
cache_current_fields!(model::AbstractModel) = error("cache_current_fields! not implemented for $(typeof(model))")

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
