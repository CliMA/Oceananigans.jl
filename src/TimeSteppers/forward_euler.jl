using Oceananigans.Fields: FunctionField, location
using Oceananigans.Utils: @apply_regionally, apply_regionally!

mutable struct ForwardEulerTimeStepper{FT, GT, IT} <: AbstractTimeStepper
                 Gⁿ :: GT
    implicit_solver :: IT
end

"""
        ForwardEulerTimeStepper(grid, prognostic_fields;
                                implicit_solver = nothing,
                                Gⁿ = map(similar, prognostic_fields))

Return a 1st-order Forward-Euler (FE) time stepper (`ForwardEulerTimeStepper`)
on `grid`, with `tracers`. The tendency fields `Gⁿ`, usually equal to 
the prognostic_fields passed as positional argument, can be specified via  optional `kwargs`.

The 1st-order Forward-Euler timestepper steps forward the state `Uⁿ` by `Δt` via

```julia
Uⁿ⁺¹ = Uⁿ + Δt * Gⁿ
```

where `Uⁿ` is the state at the ``n``-th timestep and `Gⁿ` is the tendency
at the ``n``-th timestep.
"""
function ForwardEulerTimeStepper(grid, prognostic_fields;
                                 implicit_solver::IT = nothing,
                                 Gⁿ = map(similar, prognostic_fields)) where IT

    FT = eltype(grid)
    GT = typeof(Gⁿ)

    return ForwardEulerTimeStepper{FT, GT, IT}(Gⁿ, implicit_solver)
end

reset!(timestepper::ForwardEulerTimeStepper) = nothing

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
function time_step!(model::AbstractModel{<:ForwardEulerTimeStepper}, Δt; callbacks=[])

    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0
    model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies=true)

    # If euler, then set χ = -0.5
    # Full step for tracers, fractional step for velocities.
    fe_step!(model, Δt)

    tick!(model.clock, Δt)
    model.clock.last_Δt = Δt
    model.clock.last_stage_Δt = Δt # just one stage
    
    calculate_pressure_correction!(model, Δt)
    pressure_correct_velocities!(model, Δt)

    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, Δt)

    # Return χ to initial value
    ab2_timestepper.χ = χ₀

    return nothing
end

#####
##### Time stepping in each step
#####

""" Generic implementation. """
function fe_step!(model, Δt)
    grid = model.grid
    arch = architecture(grid)
    model_fields = prognostic_fields(model)

    for (i, field) in enumerate(model_fields)
        kernel_args = (field, Δt, model.timestepper.Gⁿ[i])
        launch!(arch, grid, :xyz, fe_step_field!, kernel_args...; exclude_periphery=true)

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
Time step velocity fields via the 1st-order Forward Euler method

    `U^{n+1} = U^n + Δt G^{n} 
"""
@kernel function fe_step_field!(u, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)

    FT = typeof(χ)
    Δt = convert(FT, Δt)

    @inbounds u[i, j, k] += Δt * Gⁿ[i, j, k]
end

@kernel fe_step_field!(::FunctionField, Δt, χ, Gⁿ, G⁻) = nothing

