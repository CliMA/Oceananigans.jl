using Oceananigans.Architectures: architecture
using Oceananigans: fields

"""
    SplitRungeKutta3TimeStepper{FT, TG} <: AbstractTimeStepper

Holds parameters and tendency fields for a low storage, third-order Runge-Kutta-Wray
time-stepping scheme described by [Lan2022](@citet).
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
The tendency fields `Gⁿ` and `G⁻`, and the previous state ` Ψ⁻` can be modified via optional `kwargs`.

The scheme described by [Lan2022](@citet). In a nutshell, the 3rd-order Runge Kutta timestepper
steps forward the state `Uⁿ` by `Δt` via 3 substeps. A barotropic velocity correction step is applied
after at each substep.

The state `U` after each substep `m` is

```julia
Uᵐ⁺¹ = ζᵐ * Uⁿ + γᵐ * (Uᵐ + Δt * Gᵐ)
```

where `Uᵐ` is the state at the ``m``-th substep, `Uⁿ` is the state at the ``n``-th timestep, `Gᵐ` is the tendency
at the ``m``-th substep, and constants ``γ¹ = 1`, ``γ² = 1/4``, ``γ³ = 1/3``,
``ζ¹ = 0``, ``ζ² = 3/4``, ``ζ³ = 1/3``.

The state at the first substep is taken to be the one that corresponds to the ``n``-th timestep,
`U¹ = Uⁿ`, and the state after the third substep is then the state at the `Uⁿ⁺¹ = U³`.
"""
function SplitRungeKutta3TimeStepper(grid, prognostic_fields, args...;
                                     implicit_solver::TI = nothing,
                                     Gⁿ::TG = map(similar, prognostic_fields),
                                     Ψ⁻::PF = map(similar, prognostic_fields),
                                     G⁻::TE = nothing) where {TI, TG, PF, TE}


    @warn("Split barotropic-baroclinic time stepping with SplitRungeKutta3TimeStepper is not tested and experimental.\n" *
          "Use at own risk, and report any issues encountered.")

    !isnothing(implicit_solver) &&
        @warn("Implicit-explicit time-stepping with SplitRungeKutta3TimeStepper is not tested. " *
                "\n implicit_solver: $(typeof(implicit_solver))")

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

    split_rk3_substep!(model, Δt, nothing, nothing)
    calculate_pressure_correction!(model, Δt)
    pressure_correct_velocities!(model, Δt)
    update_state!(model, callbacks; compute_tendencies = true)

    ####
    #### Second stage
    ####

    model.clock.stage = 2

    split_rk3_substep!(model, Δt, γ², ζ²)
    calculate_pressure_correction!(model, Δt)
    pressure_correct_velocities!(model, Δt)
    update_state!(model, callbacks; compute_tendencies = true)

    ####
    #### Third stage
    ####

    model.clock.stage = 3

    split_rk3_substep!(model, Δt, γ³, ζ³)
    calculate_pressure_correction!(model, Δt)
    pressure_correct_velocities!(model, Δt)
    update_state!(model, callbacks; compute_tendencies = true)

    step_lagrangian_particles!(model, Δt)

    tick!(model.clock, Δt)
    model.clock.last_Δt = Δt

    return nothing
end

@kernel function _split_rk3_substep_field!(cⁿ, Δt, γⁿ::FT, ζⁿ, Gⁿ, cⁿ⁻¹) where FT
    i, j, k = @index(Global, NTuple)
    cⁿ[i, j, k] =  ζⁿ * cⁿ⁻¹[i, j, k] + γⁿ * (cⁿ[i, j, k] + convert(FT, Δt) * Gⁿ[i, j, k])
end

@kernel function _split_rk3_substep_field!(cⁿ, Δt, ::Nothing, ::Nothing, Gⁿ, cⁿ⁻¹)
    i, j, k = @index(Global, NTuple)
    cⁿ[i, j, k] = cⁿ[i, j, k] + Δt * Gⁿ[i, j, k]
end

function split_rk3_substep!(model, Δt, γⁿ, ζⁿ)

    grid = model.grid
    arch = architecture(grid)
    model_fields = prognostic_fields(model)

    for (i, field) in enumerate(model_fields)
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[i], model.timestepper.Ψ⁻[i])
        launch!(arch, grid, :xyz, rk3_substep_field!, kernel_args...; exclude_periphery=true)

        # TODO: function tracer_index(model, field_index) = field_index - 3, etc...
        tracer_index = Val(i - 3) # assumption

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       tracer_index,
                       model.clock,
                       stage_Δt(Δt, γⁿ, ζⁿ))
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

