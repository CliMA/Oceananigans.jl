using Oceananigans.TimeSteppers: ab2_step_field!

import Oceananigans.TimeSteppers: ab2_step!

combine_events(free_surface_event,
               velocities_events,
               tracer_events) = MultiEvent(tuple(free_surface_event, velocities_events..., tracer_events...))

combine_events(::Nothing, velocities_events, tracer_events) =
    MultiEvent(tuple(velocities_events..., tracer_events...))

combine_events(::Nothing, ::Nothing, tracer_events) = MultiEvent(tuple(tracer_events...))

function ab2_step!(model::HydrostaticFreeSurfaceModel, Δt, χ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    # Launch velocity update kernels
  
    velocities_events = []

    for name in (:u, :v)
        model.velocities isa PrescribedVelocityFields && break

        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        velocity_field = model.velocities[name]

        event = launch!(model.architecture, model.grid, :xyz, ab2_step_field!,
                        velocity_field, Δt, χ, Gⁿ, G⁻,
                        dependencies=barrier)

        push!(velocities_events, event)
    end

    # Tracer update kernels

    tracer_events = []

    for name in tracernames(model.tracers)
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        tracer_field = model.tracers[name]

        event = launch!(model.architecture, model.grid, :xyz, ab2_step_field!,
                        tracer_field, Δt, χ, Gⁿ, G⁻,
                        dependencies=barrier)

        push!(tracer_events, event)
    end

    velocities_update = MultiEvent(Tuple(velocities_events))

    # Free surface update
    
    free_surface_event = ab2_step_free_surface!(model.free_surface, velocities_update, model, Δt, χ)

    prognostic_field_events = combine_events(free_surface_event, velocities_events, tracer_events)

    wait(device(model.architecture), prognostic_field_events)

    return nothing
end
