using Oceananigans.TimeSteppers: rk3_substep_field!, stage_Δt
import Oceananigans.TimeSteppers: rk3_substep!

function rk3_substep!(model::NonhydrostaticModel, Δt, γⁿ, ζⁿ, callbacks)
    Δτ   = stage_Δt(Δt, γⁿ, ζⁿ) 
    grid = model.grid

    compute_tendencies!(model, callbacks)

    # Velocity steps
    for (i, field) in enumerate(model.velocities)
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[i], model.timestepper.G⁻[i])
        launch!(architecture(grid), grid, :xyz, rk3_substep_field!, kernel_args...; exclude_periphery=true)

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       nothing,
                       model.clock,
                       Δτ)
    end

    # Tracer steps
    for (i, name) in enumerate(propertynames(model.tracers))
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[name], model.timestepper.G⁻[name])
        launch!(architecture(grid), grid, :xyz, rk3_substep_field!, kernel_args...)

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       Val(i),
                       model.clock,
                       Δt)
    end

    compute_pressure_correction!(model, Δτ)
    make_pressure_correction!(model, Δτ)

    return nothing
end
