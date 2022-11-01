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
    @apply_regionally masking_actions!(model)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.velocities, model.tracers),  model.clock, fields(model))

    # Compute auxiliary fields
    for aux_field in model.auxiliary_fields
        compute!(aux_field)
    end

    # Calculate diffusivities
    @apply_regionally update_state_actions!(model)

    fill_halo_regions!(model.diffusivity_fields, model.clock, fields(model))
    fill_halo_regions!(model.pressures.pHY′)

    return nothing
end

# Mask immersed fields
function masking_actions!(model::NonhydrostaticModel)
    tracer_masking_events = Tuple(mask_immersed_field!(c) for c in model.tracers)
    wait(device(model.architecture), MultiEvent(tracer_masking_events))
end

function update_state_actions!(model::NonhydrostaticModel) 
    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)
    update_hydrostatic_pressure!(model)
end
