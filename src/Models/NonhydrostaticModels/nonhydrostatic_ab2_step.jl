using Oceananigans.TimeSteppers: _ab2_step_field!, implicit_step!
import Oceananigans.TimeSteppers: ab2_step!

ab2_step!(model::NonhydrostaticModel, args...) = 
    pressure_correction_ab2_step!(model, args...)

# AB2 step for NonhydrostaticModel. This is a predictor-corrector scheme where the
# predictor step for velocities is an AB2 step. The velocities are then corrected
# using the pressure correction obtained by solving a Poisson equation for the pressure. 
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

