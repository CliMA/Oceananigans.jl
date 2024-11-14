using Oceananigans.Fields: location
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: retrieve_interior_active_cells_map, retrieve_surface_active_cells_map
using Oceananigans.TimeSteppers: SplitRungeKutta3TimeStepper

import Oceananigans.TimeSteppers: split_rk3_substep!, rk3_average_pressure!, _rk3_average_pressure!

function split_rk3_substep!(model::HydrostaticFreeSurfaceModel, Δt, γⁿ, ζⁿ)
    
    rk3_substep_velocities!(model.velocities, model, Δt, γⁿ, ζⁿ)
    rk3_substep_tracers!(model.tracers, model, Δt, γⁿ, ζⁿ)
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)
    
    return nothing
end

rk3_average_pressure!(model::HydrostaticFreeSurfaceModel, γⁿ, ζⁿ) = 
    rk3_average_pressure!(model.free_surface, model.timestepper, γⁿ, ζⁿ)

function rk3_average_pressure!(free_surface::ImplicitFreeSurface, timestepper, γⁿ, ζⁿ)

    ηⁿ⁻¹ = timestepper.previous_model_fields.η    
    ηⁿ   = free_surface.η 
    
    launch!(arch, grid, _rk3_average_pressure!, ηⁿ, ηⁿ⁻¹, γⁿ, ζⁿ)
    
    return nothing
end

rk3_average_pressure!(::Nothing, args...) = nothing

function rk3_average_pressure!(free_surface::SplitExplicitFreeSurface, timestepper, γⁿ, ζⁿ)

    Uⁿ⁻¹ = timestepper.previous_model_fields.U
    Vⁿ⁻¹ = timestepper.previous_model_fields.V
    Uⁿ   = free_surface.barotropic_velocities.U
    Vⁿ   = free_surface.barotropic_velocities.V
    
    launch!(arch, grid, _rk3_average_pressure!, Uⁿ, Uⁿ⁻¹, γⁿ, ζⁿ)
    launch!(arch, grid, _rk3_average_pressure!, Vⁿ, Vⁿ⁻¹, γⁿ, ζⁿ)

    return nothing
end

#####
##### Time stepping in each substep
#####

function rk3_substep_velocities!(velocities, model, Δt, γⁿ, ζⁿ)

    for (i, name) in enumerate((:u, :v))
        Gⁿ = model.timestepper.Gⁿ[name]
        old_field = model.timestepper.previous_model_fields[name]
        velocity_field = model.velocities[name]

        launch!(model.architecture, model.grid, :xyz,
                _split_rk3_substep_field!, velocity_field, Δt, γⁿ, ζⁿ, Gⁿ, old_field)

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

rk3_substep_tracers!(::EmptyNamedTuple, model, Δt, γⁿ, ζⁿ) = nothing

function rk3_substep_tracers!(tracers, model, Δt, γⁿ, ζⁿ)

    closure = model.closure
    grid = model.grid

    # Tracer update kernels
    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        
        Gⁿ = model.timestepper.Gⁿ[tracer_name]
        old_field = model.timestepper.previous_model_fields[tracer_name]
        tracer_field = tracers[tracer_name]
        closure = model.closure

        launch!(architecture(grid), grid, :xyz,
                _split_rk3_substep_field!, tracer_field, Δt, γⁿ, ζⁿ, Gⁿ, old_field)

        implicit_step!(tracer_field,
                       model.timestepper.implicit_solver,
                       closure,
                       model.diffusivity_fields,
                       Val(tracer_index),
                       model.clock,
                       Δt)
    end

    return nothing
end