using Oceananigans.Architectures: device_event
using Oceananigans.Fields: location
using Oceananigans.TimeSteppers: ab2_step_field!
using Oceananigans.TurbulenceClosures: implicit_step!

using KernelAbstractions: NoneEvent

import Oceananigans.TimeSteppers: ab2_step!

#####
##### Step everything
#####

function ab2_step!(model::HydrostaticFreeSurfaceModel, Δt, χ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    if model.free_surface isa SplitExplicitFreeSurface
        sefs = model.free_surface
        u, v, _ = model.velocities
        barotropic_mode!(sefs.state.U, sefs.state.V, model.grid, u, v)
    end

    explicit_velocity_step_events = ab2_step_velocities!(model.velocities, model, Δt, χ)
    explicit_tracer_step_events = ab2_step_tracers!(model.tracers, model, Δt, χ)
    free_surface_event = ab2_step_free_surface!(model.free_surface, model, Δt, χ,
        MultiEvent(Tuple(explicit_velocity_step_events)))

    prognostic_field_events = MultiEvent(tuple(free_surface_event,
        explicit_velocity_step_events...,
        explicit_tracer_step_events...))

    wait(device(model.architecture), prognostic_field_events)

    return nothing
end

#####
##### Step velocities
#####

function ab2_step_velocities!(velocities, model, Δt, χ)

    barrier = device_event(model)

    # Launch velocity update kernels
    explicit_velocity_step_events = []

    for name in (:u, :v)
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        velocity_field = model.velocities[name]

        event = launch!(model.architecture, model.grid, :xyz,
                        ab2_step_field!, velocity_field, Δt, χ, Gⁿ, G⁻,
                        dependencies = device_event(model))

        push!(explicit_velocity_step_events, event)
    end

    for (i, name) in enumerate((:u, :v))
        velocity_field = model.velocities[name]

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
                       dependencies = explicit_velocity_step_events[i])
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

        event = launch!(model.architecture, model.grid, :xyz,
                        ab2_step_field!, tracer_field, Δt, χ, Gⁿ, G⁻,
                        dependencies = device_event(model))

        push!(explicit_tracer_step_events, event)
    end

    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        tracer_field = tracers[tracer_name]
        explicit_tracer_step_event = explicit_tracer_step_events[tracer_index]
        closure = model.closure

        implicit_step!(tracer_field,
                       model.timestepper.implicit_solver,
                       closure,
                       model.diffusivity_fields,
                       Val(tracer_index),
                       model.clock,
                       Δt,
                       dependencies = explicit_tracer_step_event)
    end

    return explicit_tracer_step_events
end

