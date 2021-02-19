using Oceananigans.TimeSteppers: ab2_step_field!

import Oceananigans.TimeSteppers: ab2_step!

function ab2_step!(model::HydrostaticFreeSurfaceModel, Δt, χ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    step_field_kernel! = ab2_step_field!(device(model.architecture), workgroup, worksize)

    # Launch velocity update kernels

    velocities_events = []

    for name in (:u, :v)
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        velocity_field = model.velocities[name]

        event = step_field_kernel!(velocity_field, Δt, χ, Gⁿ, G⁻,
                                   dependencies=Event(device(model.architecture)))

        push!(velocities_events, event)
    end

    # Launch tracer update kernels

    tracer_events = []

    for name in tracernames(model.tracers)
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        tracer_field = model.tracers[name]

        event = step_field_kernel!(tracer_field, Δt, χ, Gⁿ, G⁻,
                                   dependencies=Event(device(model.architecture)))

        push!(tracer_events, event)
    end

    velocities_update = MultiEvent(Tuple(velocities_events))

    # Update the free surface if not using a rigid lid once the velocities have finished updating.
    free_surface_event = ab2_step_free_surface!(model.free_surface, velocities_update, model, χ, Δt)

    wait(device(model.architecture), MultiEvent(tuple(free_surface_event, tracer_events...)))

    return nothing
end

#####
##### Free surface time-stepping: explicit, implicit, rigid lid ?
#####

function ab2_step_free_surface!(free_surface::ExplicitFreeSurface, velocities_update, model, χ, Δt)

    event = launch!(model.architecture, model.grid, :xy,
                    _ab2_step_free_surface!,
                    model.free_surface.η,
                    χ,
                    Δt,
                    model.timestepper.Gⁿ.η,
                    model.timestepper.G⁻.η,
                    dependencies=Event(device(model.architecture)))

    return event
end

@kernel function _ab2_step_free_surface!(η, χ::FT, Δt, Gηⁿ, Gη⁻) where FT
    i, j = @index(Global, NTuple)

    @inbounds begin
        η[i, j, 1] += Δt * ((FT(1.5) + χ) * Gηⁿ[i, j, 1] - (FT(0.5) + χ) * Gη⁻[i, j, 1])
    end
end


function ab2_step_free_surface!(free_surface::ImplicitFreeSurface, velocities_update, model, χ, Δt)

    ##### Implicit solver for η
    
    ## Need to wait for U* and V* to finish
    wait(device(model.architecture), velocities_update)

    return
end
