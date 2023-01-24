using Oceananigans.Architectures: device_event
using Oceananigans.Fields: location
using Oceananigans.TimeSteppers: ab2_step_field!
using Oceananigans.TurbulenceClosures: implicit_step!

import Oceananigans.TimeSteppers: ab2_step!

#####
##### Step everything
#####

function ab2_step!(model::HydrostaticFreeSurfaceModel, Δt, χ, euler = false)

    # Step locally velocity and tracers
    @apply_regionally prognostic_field_events = local_ab2_step!(model, Δt, χ, euler)

    # blocking step for implicit free surface, non blocking for explicit
    prognostic_field_events = ab2_step_free_surface!(model.free_surface, model, Δt, χ, prognostic_field_events)

    # waiting all the ab2 steps (velocities, free_surface and tracers to complete)
    @apply_regionally wait(device(model.architecture), prognostic_field_events)

    return nothing
end

function local_ab2_step!(model, Δt, χ, euler)

    explicit_velocity_step_events = ab2_step_velocities!(model.velocities, model, Δt, χ)
    explicit_tracer_step_events   = ab2_step_tracers!(model.tracers, model, Δt, χ)
    
    prognostic_field_events = (tuple(explicit_velocity_step_events...),
                               tuple(explicit_tracer_step_events...))

    return prognostic_field_events    
end

#####
##### Step velocities
#####

function ab2_step_velocities!(velocities, model, Δt, χ)

    # Launch velocity update kernels
    explicit_velocity_step_events = []

    for (i, name) in enumerate((:u, :v))
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        velocity_field = model.velocities[name]

        event = launch!(model.architecture, model.grid, :xyz,
                        ab2_step_field!, velocity_field, Δt, χ, Gⁿ, G⁻,
                        dependencies = device_event(model))

        # TODO: let next implicit solve depend on previous solve + explicit velocity step
        # Need to distinguish between solver events and tendency calculation events.
        # Note that BatchedTridiagonalSolver has a hard `wait`; this must be solved first.
        implicit_step!(velocity_field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       nothing,
                       model.clock, 
                       Δt,
                       dependencies = event)

        push!(explicit_velocity_step_events, event)
    end

    return explicit_velocity_step_events
end

#####
##### Step velocities
#####

const EmptyNamedTuple = NamedTuple{(),Tuple{}}

ab2_step_tracers!(::EmptyNamedTuple, model, Δt, χ) = [NoneEvent()]

function ab2_step_tracers!(tracers, model, Δt, χ)

    # Tracer update kernels
    explicit_tracer_step_events = []

    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        Gⁿ = model.timestepper.Gⁿ[tracer_name]
        G⁻ = model.timestepper.G⁻[tracer_name]
        tracer_field = tracers[tracer_name]
        closure = model.closure

        event = launch!(model.architecture, model.grid, :xyz,
                        ab2_step_field!, tracer_field, Δt, χ, Gⁿ, G⁻,
                        dependencies = device_event(model))

        implicit_step!(tracer_field,
                       model.timestepper.implicit_solver,
                       closure,
                       model.diffusivity_fields,
                       Val(tracer_index),
                       model.clock,
                       Δt,
                       dependencies = event)

        push!(explicit_tracer_step_events, event)
    end

    return explicit_tracer_step_events
end

