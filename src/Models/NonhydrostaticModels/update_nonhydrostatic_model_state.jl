using Oceananigans: UpdateStateCallsite
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
function update_state!(model::NonhydrostaticModel, callbacks=[]; compute_tendencies = true)
    
    # Mask immersed tracers
    foreach(model.tracers) do tracer
        @apply_regionally mask_immersed_field!(tracer)
    end

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.velocities, model.tracers), model.clock, fields(model); async = true)

    # Compute auxiliary fields
    for aux_field in model.auxiliary_fields
        compute!(aux_field)
    end

    # Calculate diffusivities and hydrostatic pressure
    @apply_regionally begin
        calculate_diffusivities!(model.diffusivity_fields, model.closure, model)
        update_hydrostatic_pressure!(model)
    end

    for callback in callbacks
        callback.callsite isa UpdateStateCallsite && callback(model)
    end

    update_biogeochemical_state!(model.biogeochemistry, model)

    compute_tendencies && 
        @apply_regionally compute_tendencies!(model, callbacks)

    return nothing
end

