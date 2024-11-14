using Oceananigans.Fields: location
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: retrieve_interior_active_cells_map, retrieve_surface_active_cells_map

import Oceananigans.TimeSteppers: split_rk3_substep!, _split_rk3_substep_field!

function split_rk3_substep!(model::HydrostaticFreeSurfaceModel, Δt, γⁿ, ζⁿ)
    
    grid         = model.grid
    timestepper  = model.timestepper
    free_surface = model.free_surface

    compute_free_surface_tendency!(grid, model, free_surface)

    rk3_substep_velocities!(model.velocities, model, Δt, γⁿ, ζⁿ)
    rk3_substep_tracers!(model.tracers, model, Δt, γⁿ, ζⁿ)

    # Full step for Implicit and Split-Explicit, substep for Explicit
    step_free_surface!(free_surface, model, timestepper, Δt)

    # Average free surface variables 
    # in the second stage
    if model.clock.stage == 2
        rk3_average_free_surface!(free_surface, grid, timestepper, γⁿ, ζⁿ)
    end
    
    return nothing
end

rk3_average_free_surface!(free_surface, args...) = nothing

function rk3_average_free_surface!(free_surface::ImplicitFreeSurface, grid, timestepper, γⁿ, ζⁿ)
    arch = architecture(grid)

    ηⁿ⁻¹ = timestepper.S⁻.η    
    ηⁿ   = free_surface.η 
    
    launch!(arch, grid, :xy, _rk3_average_free_surface!, ηⁿ, grid, ηⁿ⁻¹, γⁿ, ζⁿ)
    
    return nothing
end

function rk3_average_free_surface!(free_surface::SplitExplicitFreeSurface, grid, timestepper, γⁿ, ζⁿ)

    arch = architecture(grid)

    Uⁿ⁻¹ = timestepper.S⁻.U
    Vⁿ⁻¹ = timestepper.S⁻.V
    Uⁿ   = free_surface.barotropic_velocities.U
    Vⁿ   = free_surface.barotropic_velocities.V

    launch!(arch, grid, :xy, _rk3_average_free_surface!, Uⁿ, grid, Uⁿ⁻¹, γⁿ, ζⁿ)
    launch!(arch, grid, :xy, _rk3_average_free_surface!, Vⁿ, grid, Vⁿ⁻¹, γⁿ, ζⁿ)

    return nothing
end

@kernel function _rk3_average_free_surface!(η, grid, η⁻, γⁿ, ζⁿ) 
    i, j = @index(Global, NTuple)
    k = grid.Nz + 1
    @inbounds η[i, j, k] = ζⁿ * η⁻[i, j, k] + γⁿ * η[i, j, k] 
end

#####
##### Time stepping in each substep
#####

function rk3_substep_velocities!(velocities, model, Δt, γⁿ, ζⁿ)

    for name in (:u, :v)
        Gⁿ = model.timestepper.Gⁿ[name]
        S⁻ = model.timestepper.S⁻[name]
        velocity_field = velocities[name]

        launch!(model.architecture, model.grid, :xyz,
                _split_rk3_substep_field!, velocity_field, Δt, γⁿ, ζⁿ, Gⁿ, S⁻)

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
        S⁻ = model.timestepper.S⁻[tracer_name]
        tracer_field = tracers[tracer_name]
        closure = model.closure

        launch!(architecture(grid), grid, :xyz,
                _split_rk3_substep_field!, tracer_field, Δt, γⁿ, ζⁿ, Gⁿ, S⁻)

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