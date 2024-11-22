using Oceananigans.Fields: location
using Oceananigans.TimeSteppers: ab2_step_field!
using Oceananigans.TurbulenceClosures: implicit_step!, time_step_catke_equation!
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

    # blocking step for implicit free surface, non blocking for explicit
    ab2_step_free_surface!(model.free_surface, model, Δt, χ)

    return nothing
end

function local_ab2_step!(model, Δt, χ)
    ab2_step_tracers!(model.tracers, model, Δt, χ)
    ab2_step_velocities!(model.velocities, model, Δt, χ)
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

        # TODO: let next implicit solve depend on previous solve + explicit velocity step
        # Need to distinguish between solver events and tendency calculation events.
        # Note that BatchedTridiagonalSolver has a hard `wait`; this must be solved first.
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
##### Step Tracers (in case of CATKE, also compute TKE tendencies before updating TKE)
#####

hasclosure(closure, ClosureType) = closure isa ClosureType
hasclosure(closure_tuple::Tuple, ClosureType) = any(hasclosure(c, ClosureType) for c in closure_tuple)

const EmptyNamedTuple = NamedTuple{(),Tuple{}}

ab2_step_tracers!(::EmptyNamedTuple, model, Δt, χ) = nothing

function ab2_step_tracers!(tracers, model, Δt, χ)

    closure = model.closure
    catke_closure = hasclosure(closure, FlavorOfCATKE)

    # Tracer update kernels
    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        
        # TODO: do better than this silly criteria, also need to check closure tuples
        if catke_closure && tracer_name == :e
            time_step_catke_equation!(model)
        elseif closure isa FlavorOfTD && tracer_name == :ϵ
            @debug "Skipping AB2 step for ϵ"
        elseif closure isa FlavorOfTD && tracer_name == :e
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

