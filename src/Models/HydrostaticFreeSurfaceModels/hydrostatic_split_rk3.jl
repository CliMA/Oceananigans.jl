import Oceananigans.TimeSteppers: time_step!

@inline function rk3_substep!(model, grid, Δτ, callbacks)

    # Advancing free surface and barotropic transport velocities
    compute_momentum_tendencies!(model, callbacks)
    fill_halo_regions!(prognostic_fields(model))
    compute_free_surface_tendency!(grid, model, model.free_surface)
    fill_halo_regions!(prognostic_fields(model))
    step_free_surface!(model.free_surface, model, model.timestepper, Δτ)
    fill_halo_regions!(prognostic_fields(model))
    
    # Computing z-dependent transport velocities
    compute_transport_velocities!(model, model.free_surface)
    fill_halo_regions!(prognostic_fields(model))
    
    # compute tracer tendencies
    compute_tracer_tendencies!(model)
    fill_halo_regions!(prognostic_fields(model))
    
    # Remember to scale tracers tendencies by stretching factor
    scale_by_stretching_factor!(model.timestepper.Gⁿ, model.tracers, model.grid)
    fill_halo_regions!(prognostic_fields(model))
    
    # Finally Substep! Advance grid, tracers, and momentum
    rk3_substep_grid!(grid, model, model.vertical_coordinate, Δτ)
    rk3_substep_tracers!(model.tracers, model, Δτ)
    rk3_substep_velocities!(model.velocities, model, Δτ)

    fill_halo_regions!(prognostic_fields(model))
    # Correct for the updated barotropic mode
    make_pressure_correction!(model, Δτ)

    return nothing
end

function time_step!(model::HydrostaticFreeSurfaceModel{<:SplitRungeKutta3TimeStepper}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    cache_previous_fields!(model)
    β¹ = model.timestepper.β¹
    β² = model.timestepper.β²

    grid = model.grid

    ####
    #### First stage
    ####

    # First stage: n -> n + 1/3
    model.clock.stage = 1
    update_state!(model, callbacks)
    rk3_substep!(model, grid, Δt / β¹, callbacks)

    ####
    #### Second stage
    ####

    # Second stage: n -> n + 1/2
    model.clock.stage = 2
    update_state!(model, callbacks)
    rk3_substep!(model, grid, Δt / β², callbacks)

    ####
    #### Third stage
    ####

    # Third stage: n -> n + 1
    model.clock.stage = 3
    update_state!(model, callbacks)
    rk3_substep!(model, grid, Δt, callbacks)

    # Finalize step
    step_lagrangian_particles!(model, Δt)
    tick!(model.clock, Δt)

    return nothing
end
