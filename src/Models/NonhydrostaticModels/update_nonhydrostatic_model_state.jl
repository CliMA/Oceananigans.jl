using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.TurbulenceClosures: calculate_diffusivities!
using Oceananigans.Fields: compute!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::NonhydrostaticModel, callbacks=[])

Update peripheral aspects of the model (halo regions, diffusivities, hydrostatic
pressure) to the current model state. If `callbacks` are provided (in an array),
they are called in the end.
"""
function update_state!(model::NonhydrostaticModel, callbacks=[])
    
    # Mask immersed tracers
    tracer_masking_events = Tuple(mask_immersed_field!(c) for c in model.tracers)

    wait(device(model.architecture), MultiEvent(tracer_masking_events))

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

    [callback(model) for callback in callbacks if isa(callback.callsite, UpdateStateCallsite)]

    update_biogeochemical_state!(model.biogeochemistry, model)

    return nothing
end
