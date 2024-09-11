using Oceananigans.TimeSteppers: stage_Δt, rk3_substep_field!
import Oceananigans.TimeSteppers: rk3_substep!

function rk3_substep!(model::HydrostaticFreeSurfaceModel, Δt, γⁿ, ζⁿ)

    rk3_setup_free_surface!(model, model.free_surface, γⁿ, ζⁿ)

    workgroup, worksize = work_layout(model.grid, :xyz)
    substep_field_kernel! = rk3_substep_field!(device(architecture(model)), workgroup, worksize)
    model_fields = prognostic_fields(model)

    for (i, field) in enumerate(model_fields)
        substep_field_kernel!(field, Δt, γⁿ, ζⁿ,
                              model.timestepper.Gⁿ[i],
                              model.timestepper.G⁻[i])

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

    # blocking step for implicit free surface, non blocking for explicit
    step_free_surface!(model.free_surface, model, stage_Δt(Δt, γⁿ, ζⁿ))


    return nothing
end
