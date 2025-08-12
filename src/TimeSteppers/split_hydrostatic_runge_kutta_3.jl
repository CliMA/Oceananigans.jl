using Oceananigans.Architectures: architecture
using Oceananigans: fields

"""
    SplitRungeKutta3TimeStepper{FT, TG, TE, PF, TI} <: AbstractTimeStepper

Hold parameters and tendency fields for a low storage, third-order Runge-Kutta-Wray
time-stepping scheme described by [Lan et al. (2022)](@cite Lan2022).
"""
struct SplitRungeKutta3TimeStepper{FT, TG, TE, PF, TI} <: AbstractTimeStepper
    γ² :: FT
    γ³ :: FT
    ζ² :: FT
    ζ³ :: FT
    Gⁿ :: TG
    G⁻ :: TE # only needed for barotropic velocities in the barotropic step
    Ψ⁻ :: PF # prognostic state at the previous timestep
    implicit_solver :: TI
end

"""
    SplitRungeKutta3TimeStepper(grid, prognostic_fields, args...;
                                implicit_solver::TI = nothing,
                                Gⁿ::TG = map(similar, prognostic_fields),
                                Ψ⁻::PF = map(similar, prognostic_fields)
                                G⁻::TE = nothing) where {TI, TG, PF, TE}

Return a 3rd-order `SplitRungeKutta3TimeStepper` on `grid` and with `tracers`.
The tendency fields `Gⁿ` and `G⁻`, and the previous state ` Ψ⁻` can be modified
via optional `kwargs`.

The scheme is described by [Lan et al. (2022)](@cite Lan2022). In a nutshell,
the 3rd-order Runge-Kutta timestepper steps forward the state `Uⁿ` by `Δt` via 3 substeps.
A barotropic velocity correction step is applied after at each substep.

The state `U` after each substep `m` is

```julia
Uᵐ⁺¹ = ζᵐ * Uⁿ + γᵐ * (Uᵐ + Δt * Gᵐ)
```

where `Uᵐ` is the state at the ``m``-th substep, `Uⁿ` is the state at the ``n``-th timestep,
`Gᵐ` is the tendency at the ``m``-th substep, and constants `γ¹ = 1`, `γ² = 1/4`, `γ³ = 1/3`,
`ζ¹ = 0`, `ζ² = 3/4`, and `ζ³ = 1/3`.

The state at the first substep is taken to be the one that corresponds to the ``n``-th timestep,
`U¹ = Uⁿ`, and the state after the third substep is then the state at the `Uⁿ⁺¹ = U³`.

References
==========

Lan, R., Ju, L., Wanh, Z., Gunzburger, M., and Jones, P. (2022). High-order multirate explicit
    time-stepping schemes for the baroclinic-barotropic split dynamics in primitive equations.
    Journal of Computational Physics, 457, 111050.
"""
function SplitRungeKutta3TimeStepper(grid, prognostic_fields, args...;
                                     implicit_solver::TI = nothing,
                                     Gⁿ::TG = map(similar, prognostic_fields),
                                     Ψ⁻::PF = map(similar, prognostic_fields),
                                     G⁻::TE = nothing) where {TI, TG, PF, TE}

    @warn("Split barotropic-baroclinic time stepping with SplitRungeKutta3TimeStepper is and experimental.\n" *
          "Use at own risk, and report any issues encountered at [https://github.com/CliMA/Oceananigans.jl/issues](https://github.com/CliMA/Oceananigans.jl/issues).")

    γ² = 1 // 4
    γ³ = 2 // 3

    ζ² = 3 // 4
    ζ³ = 1 // 3

    FT = eltype(grid)

    return SplitRungeKutta3TimeStepper{FT, TG, TE, PF, TI}(γ², γ³, ζ², ζ³, Gⁿ, G⁻, Ψ⁻, implicit_solver)
end

function time_step!(model::AbstractModel{<:SplitRungeKutta3TimeStepper}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies = true)

    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    cache_previous_fields!(model)

    ####
    #### First stage
    ####

    model.clock.stage = 1

    compute_flux_bc_tendencies!(model)
    split_rk3_substep!(model, Δt, nothing, nothing)
    compute_pressure_correction!(model, Δt)
    make_pressure_correction!(model, Δt)
    update_state!(model, callbacks; compute_tendencies = true)

    ####
    #### Second stage
    ####

    model.clock.stage = 2

    compute_flux_bc_tendencies!(model)
    split_rk3_substep!(model, Δt, γ², ζ²)
    compute_pressure_correction!(model, Δt)
    make_pressure_correction!(model, Δt)
    update_state!(model, callbacks; compute_tendencies = true)

    ####
    #### Third stage
    ####

    model.clock.stage = 3

    compute_flux_bc_tendencies!(model)
    split_rk3_substep!(model, Δt, γ³, ζ³)
    compute_pressure_correction!(model, Δt)
    make_pressure_correction!(model, Δt)
    update_state!(model, callbacks; compute_tendencies = true)

    step_lagrangian_particles!(model, Δt)

    tick!(model.clock, Δt)

    return nothing
end

@kernel function _euler_substep_field!(field, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = field[i, j, k] + Δt * Gⁿ[i, j, k]
end

@kernel function _split_rk3_average_field!(field, γⁿ, ζⁿ, field⁻)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = ζⁿ * field⁻[i, j, k] + γⁿ * field[i, j, k]
end

@kernel _split_rk3_average_field!(field, ::Nothing, ::Nothing, field⁻) = nothing

function split_rk3_substep!(model, Δt, γⁿ, ζⁿ)

    grid = model.grid
    FT   = eltype(grid)
    arch = architecture(grid)
    model_fields = prognostic_fields(model)

    for (i, field) in enumerate(model_fields)
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[i], model.timestepper.Ψ⁻[i])
        launch!(arch, grid, :xyz, _euler_substep_field!, field, convert(FT, Δt), model.timestepper.Gⁿ[i])

        # TODO: function tracer_index(model, field_index) = field_index - 3, etc...
        tracer_index = Val(i - 3) # assumption

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       tracer_index,
                       model.clock,
                       Δt)

        launch!(arch, grid, :xyz, _split_rk3_average_field!, field, γⁿ, ζⁿ, model.timestepper.Ψ⁻[i])
    end
end

function cache_previous_fields!(model)

    previous_fields = model.timestepper.Ψ⁻
    model_fields = prognostic_fields(model)

    for name in keys(previous_fields)
        if !isnothing(previous_fields[name])
            parent(previous_fields[name]) .= parent(model_fields[name]) # Storing also the halos
        end
    end

    return nothing
end
