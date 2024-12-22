using Oceananigans.Fields: location
using Oceananigans.TimeSteppers: ab2_step_field!
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: time_step_catke_equation!
using Oceananigans.ImmersedBoundaries: retrieve_interior_active_cells_map, retrieve_surface_active_cells_map

import Oceananigans.TimeSteppers: ab2_step!

#####
##### Step everything
#####

function ab2_step!(model::HydrostaticFreeSurfaceModel, Δt)

    compute_free_surface_tendency!(model.grid, model, model.free_surface)

    χ = model.timestepper.χ

    # Step locally velocity and tracers
    @apply_regionally local_ab2_step!(model, Δt, χ)

    step_free_surface!(model.free_surface, model, model.timestepper, Δt)

    return nothing
end

function local_ab2_step!(model, Δt, χ)
    ab2_step_velocities!(model.velocities, model, Δt, χ)

    # Adds the fast tendency if we are not substepping. 
    # Performs the time step for `e` otherwise.
    time_step_catke_equation!(model)

    ab2_step_tracers!(model.tracers, model, Δt, χ)

    return nothing
end

#####
##### Step velocities
#####

function ab2_step_velocities!(velocities, model, Δt, χ)

    for (i, name) in enumerate((:u, :v))
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        velocity_field = model.velocities[name]

        launch!(model.architecture, model.grid, :xyz,
                ab2_step_field!, velocity_field, Δt, χ, Gⁿ, G⁻)

        implicit_step!(velocity_field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       nothing,
                       model.clock, 
                       Δt)
    end

    return nothing
end

#####
##### Step Tracers
#####

const EmptyNamedTuple = NamedTuple{(),Tuple{}}

hasclosure(closure, ClosureType) = closure isa ClosureType
hasclosure(closure_tuple::Tuple, ClosureType) = any(hasclosure(c, ClosureType) for c in closure_tuple)

ab2_step_tracers!(::EmptyNamedTuple, model, Δt, χ) = nothing

function ab2_step_tracers!(tracers, model, Δt, χ)

    closure = model.closure

    catke_in_closures = hasclosure(closure, FlavorOfCATKEWithSubsteps)
    td_in_closures    = hasclosure(closure, FlavorOfTD)

    # Tracer update kernels
    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        
        if catke_in_closures && tracer_name == :e
            @debug "Skipping AB2 step for e"
        elseif td_in_closures && tracer_name == :ϵ
            @debug "Skipping AB2 step for ϵ"
        elseif td_in_closures && tracer_name == :e
            @debug "Skipping AB2 step for e"
        else
            Gⁿ = model.timestepper.Gⁿ[tracer_name]
            G⁻ = model.timestepper.G⁻[tracer_name]
            tracer_field = tracers[tracer_name]
            closure = model.closure

            launch!(model.architecture, model.grid, :xyz,
                    ab2_step_field!, tracer_field, Δt, χ, Gⁿ, G⁻)

            implicit_step!(tracer_field,
                           model.timestepper.implicit_solver,
                           closure,
                           model.diffusivity_fields,
                           Val(tracer_index),
                           model.clock,
                           Δt)
        end
    end

    return nothing
end

