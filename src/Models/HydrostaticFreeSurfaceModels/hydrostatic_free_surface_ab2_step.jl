using Oceananigans.TimeSteppers: ab2_step_field!
using KernelAbstractions: NoneEvent

import Oceananigans.TimeSteppers: ab2_step!

implicit_vertical_viscosity_step!(u, closure, model; kwargs...) = NoneEvent()
implicit_vertical_diffusivity_step!(c, closure, model; kwargs...) = NoneEvent()

#####
##### Step everything
#####

function ab2_step!(model::HydrostaticFreeSurfaceModel, Δt, χ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    velocities_events = ab2_step_velocities!(model.velocities, model, Δt, χ)
    tracers_events = ab2_step_tracers!(model.tracers, model, Δt, χ)
    free_surface_event = ab2_step_free_surface!(model.free_surface, model, Δt, χ, MultiEvent(Tuple(velocities_events)))

    prognostic_field_events = MultiEvent(tuple(free_surface_event, velocities_events..., tracer_events...))

    wait(device(model.architecture), prognostic_field_events)

    return nothing
end

#####
##### Step velocities
#####

function ab2_step_velocities!(velocities, model, Δt, χ)

    # Launch velocity update kernels
    velocities_events = []

    for name in (:u, :v)
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        velocity_field = model.velocities[name]

        explicit_velocity_step_event = launch!(model.architecture, model.grid, :xyz,
                                               ab2_step_field!, velocity_field, Δt, χ, Gⁿ, G⁻,
                                               dependencies = barrier)

        implicit_velocity_step_event = implicit_vertical_viscosity_step!(velocity_field, model.closure, model;
                                                                         dependencies = explicit_velocity_step_event)

        push!(velocities_events, explicit_velocity_step_event, implicit_velocity_step_event)
    end

    return velocities_events
end

#####
##### Step velocities
#####

const EmptyNamedTuple = NamedTuple{(),Tuple{}}

ab2_step_tracers!(::EmptyNamedTuple, model, Δt, χ) = [NoneEvent()]

function ab2_step_tracers!(tracers, model, Δt, χ)

    # Tracer update kernels
    tracer_events = []

    for name in tracernames(tracers)
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        tracer_field = tracers[name]

        explicit_tracer_step_event = launch!(model.architecture, model.grid, :xyz,
                                             ab2_step_field!, tracer_field, Δt, χ, Gⁿ, G⁻,
                                             dependencies = barrier)

        implicit_tracer_step_event = implicit_vertical_diffusivity_step!(tracer_field, model.closure, model;
                                                                         dependencies = explicit_tracer_step_event)

        push!(tracer_events, explicit_tracer_step_event, implicit_tracer_step_event)
    end

    return tracer_events
end

