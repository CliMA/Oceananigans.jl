using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.TurbulenceClosures: calculate_diffusivities!
using Oceananigans.Fields: compute!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::NonhydrostaticModel)

Update peripheral aspects of the model (halo regions, diffusivities, hydrostatic pressure) to the current model state.
"""
function update_state!(model::NonhydrostaticModel)
    
    # Mask immersed tracers
    tracer_masking_events = Tuple(mask_immersed_field!(c) for c in model.tracers)

    wait(device(model.architecture), MultiEvent(tracer_masking_events))

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.velocities, model.tracers), model.architecture,  model.clock, fields(model))

    # Compute auxiliary fields
    for aux_field in model.auxiliary_fields
        compute!(aux_field)
    end

    # Calculate diffusivities
    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)
    fill_halo_regions!(model.diffusivity_fields, model.architecture, model.clock, fields(model))

    update_hydrostatic_pressure!(model)
    fill_halo_regions!(model.pressure.pHY′, model.architecture)

    return nothing
end
