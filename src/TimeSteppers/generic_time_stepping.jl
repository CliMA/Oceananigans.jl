#####
##### Time-stepping functionality that is independent of the TimeStepper
#####

"""
    precomputations!(diffusivities, pressures, velocities, tracers, model)

Perform precomputations necessary for an explicit timestep or substep.
"""
function precomputations!(diffusivities, pressures, velocities, tracers, model)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.velocities, model.tracers), model.architecture, 
                       model.clock, all_model_fields(model))

    # Calculate diffusivities
    calculate_diffusivities!(diffusivities, model.architecture, model.grid, model.closure,
                             model.buoyancy, velocities, tracers)

    fill_halo_regions!(model.diffusivities, model.architecture, model.clock, all_model_fields(model))

    # Calculate hydrostatic pressure
    pressure_calculation = launch!(model.architecture, model.grid, :xy, update_hydrostatic_pressure!,
                                   pressures.pHY′, model.grid, model.buoyancy, tracers,
                                   dependencies=Event(device(model.architecture)))

    # Fill halo regions for pressure
    wait(device(model.architecture), pressure_calculation)

    fill_halo_regions!(model.pressures.pHY′, model.architecture)

    return nothing
end

"""
    calculate_tendencies!(diffusivities, pressures, velocities, tracers, model)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(tendencies, velocities, tracers, pressures, diffusivities, model)

    # Note:
    #
    # "tendencies" is a NamedTuple of OffsetArrays corresponding to the tendency data for use
    # in GPU computations.
    #
    # "model.timestepper.Gⁿ" is a NamedTuple of Fields, whose data also corresponds to 
    # tendency data.
    
    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    calculate_interior_tendency_contributions!(tendencies, model.architecture, model.grid, model.advection,
                                               model.coriolis, model.buoyancy, model.surface_waves, model.closure,
                                               model.background_fields.velocities, model.background_fields.tracers,
                                               velocities, tracers, pressures.pHY′,
                                               diffusivities, model.forcing, model.clock)
                                               
    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the 
    # boundaries of the domain
    calculate_boundary_tendency_contributions!(model.timestepper.Gⁿ, model.architecture, model.velocities,
                                               model.tracers, model.clock, all_model_fields(model))

    return nothing
end

"""
    calculate_pressure_correction!(nonhydrostatic_pressure, Δt, tendencies, velocities, model)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function calculate_pressure_correction!(nonhydrostatic_pressure, Δt, velocities, model)

    fill_halo_regions!(model.velocities, model.architecture, model.clock, all_model_fields(model))

    solve_for_pressure!(nonhydrostatic_pressure, model.pressure_solver, model.architecture, model.grid, Δt, velocities)

    fill_halo_regions!(model.pressures.pNHS, model.architecture)

    return nothing
end

#####
##### Fractional and time stepping
#####

"""
Update the predictor velocities u, v, and w with the non-hydrostatic pressure via

    `u^{n+1} = u^n - δₓp_{NH} / Δx * Δt`
"""
@kernel function _pressure_correct_velocities!(U, grid, Δt, pNHS)
    i, j, k = @index(Global, NTuple)

    @inbounds U.u[i, j, k] -= ∂xᶠᵃᵃ(i, j, k, grid, pNHS) * Δt
    @inbounds U.v[i, j, k] -= ∂yᵃᶠᵃ(i, j, k, grid, pNHS) * Δt
    @inbounds U.w[i, j, k] -= ∂zᵃᵃᶠ(i, j, k, grid, pNHS) * Δt
end

"Update the solution variables (velocities and tracers)."
function pressure_correct_velocities!(U, arch, grid, Δt, pNHS)
    event = launch!(arch, grid, :xyz, _pressure_correct_velocities!, U, grid, Δt, pNHS,
                    dependencies=Event(device(arch))) 
    wait(device(arch), event)
    return nothing
end
