using Oceananigans.Architectures: architecture
using Oceananigans: fields

"""
    SplitRungeKutta3TimeStepper{FT, TG} <: AbstractTimeStepper

Holds parameters and tendency fields for a low storage, third-order Runge-Kutta-Wray
time-stepping scheme described by [LeMoin1991](@citet).
"""
struct SplitRungeKutta3TimeStepper{FT, TG, TE, PF, TI} <: AbstractTimeStepper
    γ² :: FT
    γ³ :: FT
    ζ² :: FT
    ζ³ :: FT
    Gⁿ :: TG
    G⁻ :: TE
    S⁻ :: PF # state at the previous timestep
    implicit_solver :: TI
end

"""
    SplitRungeKutta3TimeStepper(grid, tracers;
                                implicit_solver = nothing,
                                Gⁿ = TendencyFields(grid, tracers),
                                G⁻ = TendencyFields(grid, tracers))

Return a 3rd-order Runge0Kutta timestepper (`RungeKutta3TimeStepper`) on `grid` and with `tracers`.
The tendency fields `Gⁿ` and `G⁻` can be specified via  optional `kwargs`.

The scheme described by [LeMoin1991](@citet). In a nutshel, the 3rd-order
Runge Kutta timestepper steps forward the state `Uⁿ` by `Δt` via 3 substeps. A pressure correction
step is applied after at each substep.

The state `U` after each substep `m` is

```julia
Uᵐ⁺¹ = Uᵐ + Δt * (γᵐ * Gᵐ + ζᵐ * Gᵐ⁻¹)
```

where `Uᵐ` is the state at the ``m``-th substep, `Gᵐ` is the tendency
at the ``m``-th substep, `Gᵐ⁻¹` is the tendency at the previous substep,
and constants ``γ¹ = 8/15``, ``γ² = 5/12``, ``γ³ = 3/4``,
``ζ¹ = 0``, ``ζ² = -17/60``, ``ζ³ = -5/12``.

The state at the first substep is taken to be the one that corresponds to the ``n``-th timestep,
`U¹ = Uⁿ`, and the state after the third substep is then the state at the `Uⁿ⁺¹ = U⁴`.
"""
function SplitRungeKutta3TimeStepper(grid, tracers;
                                     implicit_solver::TI = nothing,
                                     Gⁿ::TG = TendencyFields(grid, tracers),
                                     G⁻::TE = TendencyFields(grid, tracers),
                                     S⁻::PF = TendencyFields(grid, tracers)) where {TI, TG, TE, PF}

    !isnothing(implicit_solver) &&
        @warn("Implicit-explicit time-stepping with RungeKutta3TimeStepper is not tested. " * 
                "\n implicit_solver: $(typeof(implicit_solver))")

    γ² = 1 // 4
    γ³ = 2 // 3

    ζ² = 3 // 4
    ζ³ = 1 // 3

    FT = eltype(grid)

    return SplitRungeKutta3TimeStepper{FT, TG, TE, PF, TI}(γ², γ³, ζ², ζ³, Gⁿ, G⁻, S⁻, implicit_solver)
end


function time_step!(model::AbstractModel{<:SplitRungeKutta3TimeStepper}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies = true)

    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    store_fields!(model)

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
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[i], model.timestepper.S⁻[i])
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

function store_fields!(model)
    
    previous_fields = model.timestepper.S⁻
    model_fields = prognostic_fields(model)
    
    for name in keys(previous_fields)
        if !isnothing(previous_fields[name])
            parent(previous_fields[name]) .= parent(model_fields[name])
        end
    end

    return nothing
end

