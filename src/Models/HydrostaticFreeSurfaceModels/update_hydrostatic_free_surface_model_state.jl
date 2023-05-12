using Oceananigans.Architectures
using Oceananigans.BoundaryConditions

using Oceananigans: UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.TurbulenceClosures: calculate_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!, inactive_node

import Oceananigans.TimeSteppers: update_state!

compute_auxiliary_fields!(auxiliary_fields) = Tuple(compute!(a) for a in auxiliary_fields)

# Note: see single_column_model_mode.jl for a "reduced" version of update_state! for
# single column models.

"""
    update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[])

Update peripheral aspects of the model (auxiliary fields, halo regions, diffusivities,
hydrostatic pressure) to the current model state. If `callbacks` are provided (in an array),
they are called in the end.
"""
update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[]) = update_state!(model, model.grid, callbacks)

function update_state!(model::HydrostaticFreeSurfaceModel, grid, callbacks)

    @apply_regionally mask_immersed_model_fields!(model, grid)

    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model))
    fill_horizontal_velocity_halos!(model.velocities.u, model.velocities.v, model.architecture)

    @apply_regionally compute_w_diffusivities_pressure!(model)

    fill_halo_regions!(model.velocities.w, model.clock, fields(model))
    fill_halo_regions!(model.diffusivity_fields, model.clock, fields(model))
    fill_halo_regions!(model.pressure.pHY′)

    for callback in callbacks
        callback.callsite isa UpdateStateCallsite && callback(model)
    end

    update_biogeochemical_state!(model.biogeochemistry, model)
    
    return nothing
end

# Mask immersed fields
function mask_immersed_model_fields!(model, grid)
    η = displacement(model.free_surface)
    fields_to_mask = merge(model.auxiliary_fields, prognostic_fields(model))

    foreach(fields_to_mask) do field
        if field !== η
            mask_immersed_field!(field)
        end
    end
    mask_immersed_field_xy!(η, k=size(grid, 3)+1, mask = inactive_node)

    return nothing
end

function compute_w_diffusivities_pressure!(model) 
    compute_w_from_continuity!(model)
    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)
    update_hydrostatic_pressure!(model.pressure.pHY′, model.architecture, model.grid, model.buoyancy, model.tracers)
    return nothing
end
