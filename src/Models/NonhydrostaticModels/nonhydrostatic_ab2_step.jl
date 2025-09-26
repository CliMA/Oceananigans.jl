using Oceananigans.TimeSteppers: ab2_step_field!, implicit_step!

# AB2 step for NonhydrostaticModel. This is a predictor-corrector scheme where the
# predictor step for velocities is an AB2 step. The velocities are then corrected
# using the pressure correction obtained by solving a Poisson equation for the pressure. 
function ab2_step!(model::NonhydrostaticModel, Δt, callbacks)
    grid = model.grid

    # Velocity steps
    for (i, field) in model.velocities
        kernel_args = (field, Δt, model.timestepper.χ, model.timestepper.Gⁿ[i], model.timestepper.G⁻[i])
        launch!(architecture(grid), grid, :xyz, ab2_step_field!, kernel_args...; exclude_periphery=true)

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       nothing,
                       model.clock,
                       Δt)
    end

    # Tracer steps
    for (i, field) in model.tracers
        kernel_args = (field, Δt, model.timestepper.χ, model.timestepper.Gⁿ[i], model.timestepper.G⁻[i])
        launch!(architecture(grid), grid, :xyz, ab2_step_field!, kernel_args...)

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       Val(i - 3),
                       model.clock,
                       Δt)
    end

    compute_pressure_correction!(model, Δt)
    make_pressure_correction!(model, Δt)

    return nothing
end
