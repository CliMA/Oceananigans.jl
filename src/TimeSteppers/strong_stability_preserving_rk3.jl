using Oceananigans.Architectures: architecture
using Oceananigans: fields

"""
    SSPRK3TimeStepper{FT, TG} <: AbstractTimeStepper

Holds parameters and tendency fields for a low storage, third-order Runge-Kutta-Wray
time-stepping scheme described by [LeMoin1991](@citet).
"""
struct SSPRK3TimeStepper{FT, TG, TI} <: AbstractTimeStepper
    γ² :: FT
    γ³ :: FT
    ζ² :: FT
    ζ³ :: FT
    Gⁿ :: TG
    previous_model_fields :: TG
    implicit_solver :: TI
end

"""
    RungeKutta3TimeStepper(grid, tracers;
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
function SSPRK3TimeStepper(grid, tracers;
                           implicit_solver::TI = nothing,
                           Gⁿ::TG = TendencyFields(grid, tracers),
                           G⁻ = TendencyFields(grid, tracers)) where {TI, TG}

    !isnothing(implicit_solver) &&
        @warn("Implicit-explicit time-stepping with RungeKutta3TimeStepper is not tested. " * 
                "\n implicit_solver: $(typeof(implicit_solver))")

    γ² = 1 // 4
    γ³ = 2 // 3

    ζ² = 3 // 4
    ζ³ = 1 // 3

    FT = eltype(grid)

    return SSPRK3TimeStepper{FT, TG, TI}(γ², γ³, ζ², ζ³, Gⁿ, G⁻, implicit_solver)
end

function time_step!(model::AbstractModel{<:SSPRK3TimeStepper}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies = true)

    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    store_old_fields!(model)

    ####
    #### First stage
    ####

    setup_free_surface!(model, model.free_surface, model.timestepper, 1)
    ssprk3_substep!(model.velocities, model.tracers, model, Δt, nothing, nothing)
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)
    pressure_correct_velocities!(model, Δt)
    update_state!(model, callbacks; compute_tendencies = true)

    ####
    #### Second stage
    ####

    setup_free_surface!(model, model.free_surface, model.timestepper, 2)
    ssprk3_substep!(model.velocities, model.tracers, model, Δt, γ², ζ²)
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)
    ssprk3_substep_free_surface!(model.free_surface, γ², ζ²)
    pressure_correct_velocities!(model, Δt)
    update_state!(model, callbacks; compute_tendencies = true)

    ####
    #### Third stage
    ####
    
    setup_free_surface!(model, model.free_surface, model.timestepper, 3)
    ssprk3_substep!(model.velocities, model.tracers, model, Δt, γ³, ζ³)
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)
    pressure_correct_velocities!(model, Δt)
  
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, Δt)

    tick!(model.clock, Δt)
    model.clock.last_Δt = Δt

    return nothing
end

function store_old_fields!(model)
    
    timestepper = model.timestepper
    previous_fields = timestepper.previous_model_fields
    new_fields = prognostic_fields(model)

    for name in keys(new_fields)
        parent(previous_fields[name]) .= parent(new_fields[name])
    end
    
    return nothing
end

@kernel function _ssprk3_substep_field!(field, Δt, γⁿ::FT, ζⁿ, Gⁿ, old_field) where FT
    i, j, k = @index(Global, NTuple)
    field[i, j, k] =  ζⁿ * old_field[i, j, k] + γⁿ * (field[i, j, k] + convert(FT, Δt) * Gⁿ[i, j, k])
end

@kernel function _ssprk3_substep_field!(field, Δt, γ¹::FT, ::Nothing, Gⁿ, old_field) where FT
    i, j, k = @index(Global, NTuple)
    field[i, j, k] = old_field[i, j, k] + convert(FT, Δt) * Gⁿ[i, j, k]
end

function ssprk3_substep_tracers!(tracers, model, Δt, γⁿ, ζⁿ)

    closure = model.closure
    grid = model.grid

    # Tracer update kernels
    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        
        Gⁿ = model.timestepper.Gⁿ[tracer_name]
        old_field = model.timestepper.previous_model_fields[tracer_name]
        tracer_field = tracers[tracer_name]
        closure = model.closure

        launch!(architecture(grid), grid, :xyz,
                _ssprk3_substep_field!, tracer_field, Δt, γⁿ, ζⁿ, Gⁿ, old_field)

        implicit_step!(tracer_field,
                       model.timestepper.implicit_solver,
                       closure,
                       model.diffusivity_fields,
                       Val(tracer_index),
                       model.clock,
                       Δt)
    end

    return nothing
end