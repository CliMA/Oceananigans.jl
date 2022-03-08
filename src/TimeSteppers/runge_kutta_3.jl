using Oceananigans.Architectures: architecture
using Oceananigans: fields

"""
    RungeKutta3TimeStepper{FT, TG} <: AbstractTimeStepper

Holds parameters and tendency fields for a low storage, third-order Runge-Kutta-Wray
time-stepping scheme described by Le and Moin (1991).
"""
struct RungeKutta3TimeStepper{FT, TG, TI} <: AbstractTimeStepper
                 γ¹ :: FT
                 γ² :: FT
                 γ³ :: FT
                 ζ² :: FT
                 ζ³ :: FT
                 Gⁿ :: TG
                 G⁻ :: TG
    implicit_solver :: TI
end

"""
    RungeKutta3TimeStepper(grid, tracers;
                           implicit_solver = nothing,
                           Gⁿ = TendencyFields(grid, tracers),
                           G⁻ = TendencyFields(grid, tracers))

Return a 3rd-order Runge0Kutta timestepper (`RungeKutta3TimeStepper`) on `grid` and with `tracers`.
The tendency fields `Gⁿ` and `G⁻` can be specified via  optional `kwargs`.

The scheme described by Le and Moin (1991) (see [LeMoin1991](@cite)). In a nutshel, the 3rd-order
Runge Kutta timestepper steps forward the state `U^n` by `Δt` via 3 substeps. A pressure correction
step is applied after at each substep.

The state `U` after each substep `m` is

```julia
Uᵐ⁺¹ = Uᵐ + Δt * (γᵐ * Gᵐ + ζᵐ * Gᵐ⁻¹)`,
```

where `Uᵐ` is the state at the ``m``-th substep, `Gᵐ` is the tendency
at the ``n``-th substep, and `Gᵐ⁻¹` is the tendency at the previous
substep, and constants ``γ¹ = 8/15``, ``γ² = 5/12``, ``γ³ = 3/4``,
``ζ¹ = 0``, ``ζ² = -17/60``, ``ζ³ = -5/12``.

The state at the first substep is taken to be the one that corresponds to the ``n``-th timestep,
`U¹ = Uⁿ`, and the state after the third substep is then the state at the `Uⁿ⁺¹ = U⁴`.
"""
function RungeKutta3TimeStepper(grid, tracers;
                                implicit_solver::TI = nothing,
                                Gⁿ::TG = TendencyFields(grid, tracers),
                                G⁻ = TendencyFields(grid, tracers)) where {TI, TG}

    !isnothing(implicit_solver) &&
        @warn("Implicit-explicit time-stepping with RungeKutta3TimeStepper is not tested. " * 
              "\n implicit_solver: $(typeof(implicit_solver))")

    γ¹ = 8 // 15
    γ² = 5 // 12
    γ³ = 3 // 4

    ζ² = -17 // 60
    ζ³ = -5 // 12

    FT = eltype(grid)

    return RungeKutta3TimeStepper{FT, TG, TI}(γ¹, γ², γ³, ζ², ζ³, Gⁿ, G⁻, implicit_solver)
end

#####
##### Time steppping
#####

"""
    time_step!(model::AbstractModel{<:RungeKutta3TimeStepper}, Δt)

Step forward `model` one time step `Δt` with a 3rd-order Runge-Kutta method.
The 3rd-order Runge-Kutta method takes three intermediate substep stages to
achieve a single timestep. A pressure correction step is applied at each intermediate
stage.
"""
function time_step!(model::AbstractModel{<:RungeKutta3TimeStepper}, Δt)
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model)

    γ¹ = model.timestepper.γ¹
    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    first_stage_Δt  = γ¹ * Δt
    second_stage_Δt = (γ² + ζ²) * Δt
    third_stage_Δt  = (γ³ + ζ³) * Δt

    #
    # First stage
    #

    calculate_tendencies!(model)

    correct_immersed_tendencies!(model, Δt, γ¹, 0)

    rk3_substep!(model, Δt, γ¹, nothing)

    calculate_pressure_correction!(model, first_stage_Δt)
    pressure_correct_velocities!(model, first_stage_Δt)

    tick!(model.clock, first_stage_Δt; stage=true)
    store_tendencies!(model)
    update_state!(model)
    update_particle_properties!(model, first_stage_Δt)

    #
    # Second stage
    #

    calculate_tendencies!(model)

    correct_immersed_tendencies!(model, Δt, γ², ζ²)

    rk3_substep!(model, Δt, γ², ζ²)

    calculate_pressure_correction!(model, second_stage_Δt)
    pressure_correct_velocities!(model, second_stage_Δt)

    tick!(model.clock, second_stage_Δt; stage=true)
    store_tendencies!(model)
    update_state!(model)
    update_particle_properties!(model, second_stage_Δt)

    #
    # Third stage
    #

    calculate_tendencies!(model)
    
    correct_immersed_tendencies!(model, Δt, γ³, ζ³)

    rk3_substep!(model, Δt, γ³, ζ³)

    calculate_pressure_correction!(model, third_stage_Δt)
    pressure_correct_velocities!(model, third_stage_Δt)

    tick!(model.clock, third_stage_Δt)
    update_state!(model)
    update_particle_properties!(model, third_stage_Δt)

    return nothing
end

#####
##### Time stepping in each substep
#####

stage_Δt(Δt, γⁿ, ζⁿ) = Δt * (γⁿ + ζⁿ)
stage_Δt(Δt, γⁿ, ::Nothing) = Δt * γⁿ

function rk3_substep!(model, Δt, γⁿ, ζⁿ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(architecture(model)))

    substep_field_kernel! = rk3_substep_field!(device(architecture(model)), workgroup, worksize)

    model_fields = prognostic_fields(model)

    events = []

    for (i, field) in enumerate(model_fields)

        field_event = substep_field_kernel!(field, Δt, γⁿ, ζⁿ,
                                            model.timestepper.Gⁿ[i],
                                            model.timestepper.G⁻[i],
                                            dependencies=barrier)

        # TODO: function tracer_index(model, field_index) = field_index - 3, etc...
        tracer_index = i - 3 # assumption

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.clock,
                       stage_Δt(Δt, γⁿ, ζⁿ),
                       model.closure,
                       tracer_index,
                       model.diffusivity_fields,
                       model.tracers,
                       dependencies = field_event)

        push!(events, field_event)
    end

    wait(device(architecture(model)), MultiEvent(Tuple(events)))

    return nothing
end

"""
Time step fields via the 3rd-order Runge-Kutta method

    `U^{m+1} = U^m + Δt (γⁿ G^{m} + ζⁿ G^{m-1})`,

where `m` denotes the substage.
"""

"""
Time step velocity fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_field!(U, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[i, j, k] += Δt * (γⁿ * Gⁿ[i, j, k] + ζⁿ * G⁻[i, j, k])
    end
end

"""
Time step velocity fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_field!(U, Δt, γ¹, ::Nothing, G¹, G⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[i, j, k] += Δt * γ¹ * G¹[i, j, k]
    end
end
