using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.TurbulenceClosures: calculate_diffusivities!
using Oceananigans.Fields: compute!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::NonhydrostaticModel)

Update peripheral aspects of the model (halo regions, diffusivities, hydrostatic pressure) to the current model state.
"""
function update_state!(model::NonhydrostaticModel)
    
    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.velocities, model.tracers),  model.clock, fields(model))

    # Compute auxiliary fields
    for aux_field in model.auxiliary_fields
        compute!(aux_field)
    end

    # Calculate diffusivities
    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)
    fill_halo_regions!(model.diffusivity_fields, model.clock, fields(model))

    update_hydrostatic_pressure!(model)
    fill_halo_regions!(model.pressures.pHY′)

    return nothing
end
