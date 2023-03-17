using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
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
function update_state!(model::NonhydrostaticModel, callbacks=[]; compute_tendencies = true)
    
    # Mask immersed tracers
    foreach(mask_immersed_field!, model.tracers)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.velocities, model.tracers))

    # Compute auxiliary fields
    for aux_field in model.auxiliary_fields
        compute!(aux_field)
    end

    # Calculate diffusivities
    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)
    fill_halo_regions!(model.diffusivity_fields)

    update_hydrostatic_pressure!(model)
    fill_halo_regions!(model.pressures.pHY′)

    [callback(model) for callback in callbacks if isa(callback.callsite, UpdateStateCallsite)]

    compute_tendencies && 
        @apply_regionally compute_tendencies!(model, callbacks)

    return nothing
end
