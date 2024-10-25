using Oceananigans.Fields: location
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: retrieve_interior_active_cells_map, retrieve_surface_active_cells_map
using Oceananigans.TimeSteppers: SSPRK3TimeStepper

import Oceananigans.TimeSteppers: time_step!

function time_step!(model::AbstractModel{<:SSPRK3TimeStepper}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies = true)

    γ¹ = model.timestepper.γ¹
    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    store_old_fields!(model)

    ####
    #### First stage
    ####

    setup_free_surface!(model, model.free_surface, model.timestepper, 1)
    ssprk3_substep_velocities!(model.velocities, model, Δt, nothing, nothing)
    ssprk3_substep_tracers!(model.tracers, model, Δt, nothing, nothing)
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)
    pressure_correct_velocities!(model, Δt)
    update_state!(model, callbacks; compute_tendencies = true)

    ####
    #### Second stage
    ####

    setup_free_surface!(model, model.free_surface, model.timestepper, 2)
    ssprk3_substep_velocities!(model.velocities, model, Δt, γ², ζ²)
    ssprk3_substep_tracers!(model.tracers, model, Δt, γ², ζ²)
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)
    ssprk3_substep_free_surface!(model.free_surface, γ², ζ²)
    pressure_correct_velocities!(model, Δt)

    update_state!(model, callbacks; compute_tendencies = true)

    ####
    #### Third stage
    ####
    
    setup_free_surface!(model, model.free_surface, model.timestepper, 3)
    ssprk3_substep_velocities!(model.velocities, model, Δt, γ³, ζ³)
    ssprk3_substep_tracers!(model.tracers, model, Δt, γ³, ζ³)
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)
    pressure_correct_velocities!(model, Δt)
  
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, Δt)

    tick!(model.clock, Δt)
    model.clock.last_Δt = Δt

    return nothing
end

function store_old_fields!(model::HydrostaticFreeSurfaceModel)
    
    timestepper = model.timestepper
    previous_fields   = timestepper.previous_model_fields
    new_fields = prognostic_fields(model)

    for name in keys(new_fields)
        parent(previous_fields[name]) .= parent(new_fields[name])
    end
    
    Uᵐ = model.free_surface.state.Uᵐ⁻²
    Vᵐ = model.free_surface.state.Vᵐ⁻²

    U̅ = model.free_surface.state.U̅
    V̅ = model.free_surface.state.V̅

    parent(Uᵐ) .= parent(U̅)
    parent(Vᵐ) .= parent(V̅)

    return nothing
end

function ssprk3_substep_free_surface!(free_surface, γⁿ, ζⁿ)

    Uᵐ = free_surface.state.Uᵐ⁻²
    Vᵐ = free_surface.state.Vᵐ⁻²

    U̅ = free_surface.state.U̅
    V̅ = free_surface.state.V̅
    
    if !isnothing(ζⁿ)
        parent(U̅) .= ζⁿ * parent(Uᵐ) + γⁿ * parent(U̅)
        parent(V̅) .= ζⁿ * parent(Vᵐ) + γⁿ * parent(V̅)
    end
    
    return nothing
end

#####
##### Time stepping in each substep
#####

@kernel function _ssprk3_substep_field!(field, Δt, γⁿ, ζⁿ, Gⁿ, old_field)
    i, j, k = @index(Global, NTuple)
    field[i, j, k] =  ζⁿ * old_field[i, j, k] + γⁿ * (field[i, j, k] + Δt * Gⁿ[i, j, k])
end

@kernel function _ssprk3_substep_field!(field, Δt, ::Nothing, ::Nothing, Gⁿ, old_field)
    i, j, k = @index(Global, NTuple)
    field[i, j, k] = old_field[i, j, k] + Δt * Gⁿ[i, j, k]
end

function ssprk3_substep_velocities!(velocities, model, Δt, γⁿ, ζⁿ)

    for (i, name) in enumerate((:u, :v))
        Gⁿ = model.timestepper.Gⁿ[name]
        old_field = model.timestepper.previous_model_fields[name]
        velocity_field = model.velocities[name]

        launch!(model.architecture, model.grid, :xyz,
                _ssprk3_substep_field!, velocity_field, Δt, γⁿ, ζⁿ, Gⁿ, old_field)

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

ssprk3_substep_tracers!(::EmptyNamedTuple, model, Δt, γⁿ, ζⁿ) = nothing

function ssprk3_substep_tracers!(tracers, model, Δt, γⁿ, ζⁿ)

    closure = model.closure
    grid = model.grid

    # Tracer update kernels
    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        
        Gⁿ = model.timestepper.Gⁿ[tracer_name]
        old_field = model.timestepper.previous_model_fields[tracer_name]
        tracer_field = tracers[tracer_name]
        closure = model.closure

        launch!(architecture(grid), grid, :xyz,
                _ssprk3_substep_field!, tracer_field, Δt, γⁿ, ζⁿ, Gⁿ, old_field)

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

