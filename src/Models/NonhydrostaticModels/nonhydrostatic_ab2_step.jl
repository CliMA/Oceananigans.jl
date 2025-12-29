using Oceananigans.TimeSteppers: _ab2_step_field!, implicit_step!
import Oceananigans.TimeSteppers: ab2_step!

"""
    ab2_step!(model::NonhydrostaticModel, Δt, callbacks)

Advance `NonhydrostaticModel` by one Adams-Bashforth 2nd-order time step with pressure correction.
Dispatches to `pressure_correction_ab2_step!` which implements a predictor-corrector scheme
"""
ab2_step!(model::NonhydrostaticModel, args...) = 
    pressure_correction_ab2_step!(model, args...)

"""
    pressure_correction_ab2_step!(model, Δt, callbacks)

Implement the AB2 time step with pressure correction for `NonhydrostaticModel`.

This predictor-corrector scheme:
1. Computes tendencies `Gⁿ` for all prognostic fields
2. Advances velocities: `u* = uⁿ + Δt * AB2(Gᵤ)` (predictor step)
3. Advances tracers: `cⁿ⁺¹ = cⁿ + Δt * AB2(Gᶜ)`
4. Applies implicit vertical diffusion (if configured)
5. Solves `∇²p = ∇·u* / Δt` for pressure correction
6. Corrects velocities: `uⁿ⁺¹ = u* - Δt * ∇p` to satisfy `∇·uⁿ⁺¹ = 0`
"""
function pressure_correction_ab2_step!(model, Δt, callbacks)
    grid = model.grid

    compute_tendencies!(model, callbacks)

    # Velocity steps
    for (i, field) in enumerate(model.velocities)
        kernel_args = (field, Δt, model.timestepper.χ, model.timestepper.Gⁿ[i], model.timestepper.G⁻[i])
        launch!(architecture(grid), grid, :xyz, _ab2_step_field!, kernel_args...; exclude_periphery=true)

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       nothing,
                       model.clock,
                       fields(model),
                       Δt)
    end

    # Tracer steps
    for (i, name) in enumerate(propertynames(model.tracers))
        field = model.tracers[name]
        kernel_args = (field, Δt, model.timestepper.χ, model.timestepper.Gⁿ[name], model.timestepper.G⁻[name])
        launch!(architecture(grid), grid, :xyz, _ab2_step_field!, kernel_args...)

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       Val(i),
                       model.clock,
                       fields(model),
                       Δt)
    end

    compute_pressure_correction!(model, Δt)
    make_pressure_correction!(model, Δt)

    return nothing
end

