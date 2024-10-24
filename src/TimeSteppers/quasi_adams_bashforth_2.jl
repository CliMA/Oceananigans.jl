using Oceananigans.Fields: FunctionField, location
using Oceananigans.Utils: @apply_regionally, apply_regionally!

mutable struct QuasiAdamsBashforth2TimeStepper{FT, GT, IT} <: AbstractTimeStepper
                  χ :: FT
                 Gⁿ :: GT
                 G⁻ :: GT
    implicit_solver :: IT
end

"""
    QuasiAdamsBashforth2TimeStepper(grid, tracers,
                                    χ = 0.1;
                                    implicit_solver = nothing,
                                    Gⁿ = TendencyFields(grid, tracers),
                                    G⁻ = TendencyFields(grid, tracers))

Return a 2nd-order quasi Adams-Bashforth (AB2) time stepper (`QuasiAdamsBashforth2TimeStepper`)
on `grid`, with `tracers`, and AB2 parameter `χ`. The tendency fields `Gⁿ` and `G⁻` can be
specified via  optional `kwargs`.

The 2nd-order quasi Adams-Bashforth timestepper steps forward the state `Uⁿ` by `Δt` via

```julia
Uⁿ⁺¹ = Uⁿ + Δt * [(3/2 + χ) * Gⁿ - (1/2 + χ) * Gⁿ⁻¹]
```

where `Uⁿ` is the state at the ``n``-th timestep, `Gⁿ` is the tendency
at the ``n``-th timestep, and `Gⁿ⁻¹` is the tendency at the previous
timestep (`G⁻`).

!!! note "First timestep"
    For the first timestep, since there are no saved tendencies from the previous timestep,
    the `QuasiAdamsBashforth2TimeStepper` performs an Euler timestep:

    ```julia
    Uⁿ⁺¹ = Uⁿ + Δt * Gⁿ
    ```
"""
function QuasiAdamsBashforth2TimeStepper(grid, tracers,
                                         χ = 0.1;
                                         implicit_solver::IT = nothing,
                                         Gⁿ = TendencyFields(grid, tracers),
                                         G⁻ = TendencyFields(grid, tracers)) where IT

    FT = eltype(grid)
    GT = typeof(Gⁿ)
    χ  = convert(FT, χ)

    return QuasiAdamsBashforth2TimeStepper{FT, GT, IT}(χ, Gⁿ, G⁻, implicit_solver)
end

reset!(timestepper::QuasiAdamsBashforth2TimeStepper) = nothing

#####
##### Time steppping
#####

"""
    time_step!(model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt; euler=false)

Step forward `model` one time step `Δt` with a 2nd-order Adams-Bashforth method and
pressure-correction substep. Setting `euler=true` will take a forward Euler time step.
The tendencies are calculated by the `update_step!` at the end of the `time_step!` function.

The steps of the Quasi-Adams-Bashforth second-order (AB2) algorithm are:

1. If this the first time step (`model.clock.iteration == 0`), then call `update_state!` and calculate the tendencies.
2. Advance tracers in time and compute predictor velocities (including implicit vertical diffusion).
3. Solve the elliptic equation for pressure (three dimensional for the non-hydrostatic model, two-dimensional for the hydrostatic model).
4. Correct the velocities based on the results of step 3.
5. Store the old tendencies.
6. Update the model state.
7. Compute tendencies for the next time step
"""
function time_step!(model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt;
                    callbacks=[], euler=false)

    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0
    model.clock.iteration == 0 && update_state!(model, callbacks)

    ab2_timestepper = model.timestepper

    # Change the default χ if necessary, which occurs if:
    #   * We detect that the time-step size has changed.
    #   * We detect that this is the "first" time-step, which means we
    #     need to take an euler step. Note that model.clock.last_Δt is
    #     initialized as Inf
    #   * The user has passed euler=true to time_step!
    euler = euler || (Δt != model.clock.last_Δt)
    
    # If euler, then set χ = -0.5
    minus_point_five = convert(eltype(model.grid), -0.5)
    χ = ifelse(euler, minus_point_five, ab2_timestepper.χ)

    # Set time-stepper χ (this is used in ab2_step!, but may also be used elsewhere)
    χ₀ = ab2_timestepper.χ # Save initial value
    ab2_timestepper.χ = χ

    # Ensure zeroing out all previous tendency fields to avoid errors in
    # case G⁻ includes NaNs. See https://github.com/CliMA/Oceananigans.jl/issues/2259
    if euler
        @debug "Taking a forward Euler step."
        for field in ab2_timestepper.G⁻
            !isnothing(field) && @apply_regionally fill!(field, 0)
        end
    end

    # Be paranoid and update state at iteration 0
    model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies=true)
    
    ab2_step!(model, Δt) # full step for tracers, fractional step for velocities.
    
    tick!(model.clock, Δt)
    model.clock.last_Δt = Δt
    model.clock.last_stage_Δt = Δt # just one stage
    
    calculate_pressure_correction!(model, Δt)
    @apply_regionally correct_velocities_and_store_tendencies!(model, Δt)

    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, Δt)

    # Return χ to initial value
    ab2_timestepper.χ = χ₀
    
    return nothing
end

function correct_velocities_and_store_tendencies!(model, Δt)
    pressure_correct_velocities!(model, Δt)
    store_tendencies!(model)
    return nothing
end

#####
##### Time stepping in each step
#####

""" Generic implementation. """
function ab2_step!(model, Δt)

    grid = model.grid
    arch = architecture(grid)
    model_fields = prognostic_fields(model)
    χ = model.timestepper.χ

    for (i, field) in enumerate(model_fields)
        kernel_args = (field, Δt, χ, model.timestepper.Gⁿ[i], model.timestepper.G⁻[i])
        launch!(arch, grid, :xyz, ab2_step_field!, kernel_args...; exclude_periphery=true)

        # TODO: function tracer_index(model, field_index) = field_index - 3, etc...
        tracer_index = Val(i - 3) # assumption

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       tracer_index,
                       model.clock,
                       Δt)
    end

    return nothing
end

"""
Time step velocity fields via the 2nd-order quasi Adams-Bashforth method

    `U^{n+1} = U^n + Δt ((3/2 + χ) * G^{n} - (1/2 + χ) G^{n-1})`

"""
@kernel function ab2_step_field!(u, Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    FT = eltype(χ)
    one_point_five = convert(FT, 1.5)
    oh_point_five  = convert(FT, 0.5)

    @inbounds u[i, j, k] += convert(FT, Δt) * ((one_point_five + χ) * Gⁿ[i, j, k] - (oh_point_five + χ) * G⁻[i, j, k])
end

@kernel ab2_step_field!(::FunctionField, Δt, χ, Gⁿ, G⁻) = nothing

