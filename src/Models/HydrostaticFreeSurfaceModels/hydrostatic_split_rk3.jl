import Oceananigans.TimeSteppers: time_step!

@inline function rk3_substep(model, grid, Δτ)

    # Advancing free surface and barotropic transport velocities
    compute_hydrostatic_momentum_tendencies!(model, model.velocities, :xyz)
    compute_momentum_flux_bcs!(model)
    compute_free_surface_tendency!(grid, model, model.free_surface)
    step_free_surface!(model.free_surface, model, model.timestepper, Δτ)

    # Computing z-dependent transport velocities
    fill_halo_regions!(prognostic_fields(model))
    fill_halo_regions!(model.transport_velocities)
    update_vertical_velocities!(model.transport_velocities, grid, model)
    fill_halo_regions!(model.transport_velocities)

    # Advancing tracers, momentum, and grid
    compute_hydrostatic_tracer_tendencies!(model, :xyz)
    compute_tracer_flux_bcs!(model)
    scale_by_stretching_factor!(model.timestepper.Gⁿ, model.tracers, model.grid)
    rk3_substep_grid!(grid, model, model.vertical_coordinate, Δτ)
    rk3_substep_tracers!(model.tracers, model, Δτ)
    rk3_substep_velocities!(model.velocities, model, Δτ)

    # Update state (substitute new barotropic velocities and compute pressure gradient)
    make_pressure_correction!(model, Δτ)
    update_state!(model, callbacks; compute_tendencies=false)

    return nothing
end

function time_step!(model::AbstractModel{<:SplitRungeKutta3TimeStepper}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies=false)

    cache_previous_fields!(model)
    β¹ = model.timestepper.β¹
    β² = model.timestepper.β²

    grid = model.grid

    ####
    #### First stage
    ####

    # First stage: n -> n + 1/3
    model.clock.stage = 1
    Δτ = Δt / β¹
    rk3_substep!(model, grid, Δτ)

    ####
    #### Second stage
    ####

    # Second stage: n -> n + 1/2
    model.clock.stage = 2
    Δτ = Δt / β²
    rk3_substep!(model, grid, Δτ)

    ####
    #### Third stage
    ####

    # Third stage: n -> n + 1
    model.clock.stage = 3
    rk3_substep!(model, grid, Δt)

    # Finalize step
    step_lagrangian_particles!(model, Δt)
    tick!(model.clock, Δt)

    return nothing
end
