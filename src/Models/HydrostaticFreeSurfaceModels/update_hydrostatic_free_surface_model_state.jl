using Oceananigans.Architectures
using Oceananigans.Architectures: device_event
using Oceananigans.BoundaryConditions
using Oceananigans.TurbulenceClosures: calculate_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_reduced_field_xy!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!

import Oceananigans.TimeSteppers: update_state!

compute_auxiliary_fields!(auxiliary_fields) = Tuple(compute!(a) for a in auxiliary_fields)

# Note: see single_column_model_mode.jl for a "reduced" version of update_state! for
# single column models.

"""
    update_state!(model::HydrostaticFreeSurfaceModel)

Update peripheral aspects of the model (auxiliary fields, halo regions, diffusivities,
hydrostatic pressure) to the current model state. If `callbacks` are provided (in an array),
they are called in the end.
"""
update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[]) = update_state!(model, model.grid, callbacks)

function update_state!(model::HydrostaticFreeSurfaceModel, grid, callbacks)

    @apply_regionally masking_actions!(model)

    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model))
    fill_horizontal_velocity_halos!(model.velocities.u, model.velocities.v, model.architecture)

    @apply_regionally compute_w_diffusivities_pressure!(model)

    fill_halo_regions!(model.velocities.w, model.clock, fields(model))
    fill_halo_regions!(model.diffusivity_fields, model.clock, fields(model))
    fill_halo_regions!(model.pressure.pHY′)

    [callback(model) for callback in callbacks if isa(callback.callsite, UpdateStateCallsite)]
    
    return nothing
end

# Mask immersed fields
function masking_actions!(model)
    η = displacement(model.free_surface)
    masking_events = Any[mask_immersed_field!(field)
                         for field in merge(model.auxiliary_fields, prognostic_fields(model)) if field !== η]
    push!(masking_events, mask_immersed_reduced_field_xy!(η, k=size(model.grid, 3)))    
    wait(device(model.architecture), MultiEvent(Tuple(masking_events)))
end

function compute_w_diffusivities_pressure!(model) 
    compute_w_from_continuity!(model)
    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)
    update_hydrostatic_pressure!(model.pressure.pHY′, model.architecture, model.grid, model.buoyancy, model.tracers)

    return nothing
end
