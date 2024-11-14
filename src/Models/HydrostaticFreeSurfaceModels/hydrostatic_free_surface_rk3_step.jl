using Oceananigans.Fields: location
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: retrieve_interior_active_cells_map, retrieve_surface_active_cells_map

import Oceananigans.TimeSteppers: split_rk3_substep!, _split_rk3_substep_field!, rk3_average_pressure!, _rk3_average_pressure!

function split_rk3_substep!(model::HydrostaticFreeSurfaceModel, Δt, γⁿ, ζⁿ)
    
    compute_free_surface_tendency!(model.grid, model, model.free_surface)
    rk3_substep_velocities!(model.velocities, model, Δt, γⁿ, ζⁿ)
    rk3_substep_tracers!(model.tracers, model, Δt, γⁿ, ζⁿ)
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)
    
    return nothing
end

rk3_average_pressure!(model::HydrostaticFreeSurfaceModel, γⁿ, ζⁿ) = 
    rk3_average_pressure!(model.grid, model.free_surface, model.timestepper, γⁿ, ζⁿ)

function rk3_average_pressure!(grid, free_surface::ImplicitFreeSurface, timestepper, γⁿ, ζⁿ)

    arch = architecture(grid)

    ηⁿ⁻¹ = timestepper.previous_model_fields.η    
    ηⁿ   = free_surface.η 
    
    Nx, Ny, _ = size(grid)
    params = KernelParameters(1:Nx, 1:Ny)

    launch!(arch, grid, params, _rk3_average_free_surface!, parent(ηⁿ), parent(ηⁿ⁻¹), γⁿ, ζⁿ)
    
    return nothing
end

rk3_average_pressure!(::Nothing, args...) = nothing

function rk3_average_pressure!(grid, free_surface::SplitExplicitFreeSurface, timestepper, γⁿ, ζⁿ)

    arch = architecture(grid)

    Uⁿ⁻¹ = timestepper.previous_model_fields.U
    Vⁿ⁻¹ = timestepper.previous_model_fields.V
    Uⁿ   = free_surface.barotropic_velocities.U
    Vⁿ   = free_surface.barotropic_velocities.V

    Nx, Ny, _ = size(grid)

    launch!(arch, grid, (Nx, Ny), _rk3_average_free_surface!, Uⁿ, Uⁿ⁻¹, γⁿ, ζⁿ)
    launch!(arch, grid, (Nx, Ny), _rk3_average_free_surface!, Vⁿ, Vⁿ⁻¹, γⁿ, ζⁿ)

    return nothing
end

@kernel function _rk3_average_free_surface!(pressure, old_pressure, γⁿ, ζⁿ) 
    i, j = @index(Global, NTuple)
    pressure[i, j, k] = γⁿ * pressure[i, j, k] + ζⁿ * old_pressure[i, j, k]
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