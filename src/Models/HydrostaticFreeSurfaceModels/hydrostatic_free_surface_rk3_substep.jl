using Oceananigans.TimeSteppers: stage_Δt, rk3_substep_field!
import Oceananigans.TimeSteppers: rk3_substep!

rk3_setup_free_surface!(model, free_surface, γⁿ, ζⁿ) = nothing

function rk3_substep!(model::HydrostaticFreeSurfaceModel, Δt, γⁿ, ζⁿ)

    rk3_setup_free_surface!(model, model.free_surface, γⁿ, ζⁿ)
    
    for (i, name) in enumerate((:u, :v))
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        velocity_field = model.velocities[name]

        launch!(model.architecture, model.grid, :xyz,
                rk3_substep_field!, velocity_field, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)

        # TODO: let next implicit solve depend on previous solve + explicit velocity step
        # Need to distinguish between solver events and tendency calculation events.
        # Note that BatchedTridiagonalSolver has a hard `wait`; this must be solved first.
        implicit_step!(velocity_field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       nothing,
                       model.clock, 
                       stage_Δt(Δt, γⁿ, ζⁿ))
    end

    # Tracer update kernels
    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))
        
        # TODO: do better than this silly criteria, also need to check closure tuples
        Gⁿ = model.timestepper.Gⁿ[tracer_name]
        G⁻ = model.timestepper.G⁻[tracer_name]
        tracer_field = model.tracers[tracer_name]
        closure = model.closure

        launch!(model.architecture, model.grid, :xyz,
                rk3_substep_field!, tracer_field, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)
                
        implicit_step!(tracer_field,
                       model.timestepper.implicit_solver,
                       closure,
                       model.diffusivity_fields,
                       Val(tracer_index),
                       model.clock,
                       stage_Δt(Δt, γⁿ, ζⁿ))
    end

    # blocking step for implicit free surface, non blocking for explicit
    step_free_surface!(model.free_surface, model, stage_Δt(Δt, γⁿ, ζⁿ))

    return nothing
end
