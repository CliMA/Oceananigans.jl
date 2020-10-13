"""
    precomputations!(model)

Perform precomputations necessary for an explicit timestep or substep.
"""
function precomputations!(model)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.velocities, model.tracers), model.architecture, 
                       model.clock, all_model_fields(model))

    # Calculate diffusivities
    calculate_diffusivities!(model.diffusivities, model.architecture, model.grid, model.closure,
                             model.buoyancy, model.velocities, model.tracers)

    fill_halo_regions!(model.diffusivities, model.architecture, model.clock, all_model_fields(model))

    # Calculate hydrostatic pressure
    pressure_calculation = launch!(model.architecture, model.grid, :xy, update_hydrostatic_pressure!,
                                   model.pressures.pHY′, model.grid, model.buoyancy, model.tracers,
                                   dependencies=Event(device(model.architecture)))

    # Fill halo regions for pressure
    wait(device(model.architecture), pressure_calculation)

    fill_halo_regions!(model.pressures.pHY′, model.architecture)

    return nothing
end

