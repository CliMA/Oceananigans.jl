using Oceananigans.Architectures
using Oceananigans.BoundaryConditions

using Oceananigans: UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: update_boundary_conditions!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!, inactive_node
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!, p_kernel_parameters

import Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!
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
update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[]) =  update_state!(model, model.grid, callbacks)

function update_state!(model::HydrostaticFreeSurfaceModel, grid, callbacks)

    arch = architecture(grid)

    @apply_regionally begin
        mask_immersed_model_fields!(model, grid)
        update_model_field_time_series!(model, model.clock)
        update_boundary_conditions!(fields(model), model)
    end
    
    # Fill the halos
    fill_halo_regions!(prognostic_fields(model), model.grid, model.clock, fields(model); async=true)

    w_parameters = w_kernel_parameters(model.grid)
    p_parameters = p_kernel_parameters(model.grid)

    update_vertical_velocities!(model.velocities, model.grid, model; parameters = w_parameters)
    update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers; parameters = p_parameters)
    compute_diffusivities!(model.diffusivity_fields, model.closure, model; parameters = :xyz)

    fill_halo_regions!(model.diffusivity_fields; only_local_halos=true)

    [callback(model) for callback in callbacks if callback.callsite isa UpdateStateCallsite]

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
    mask_immersed_field_xy!(η, k=size(grid, 3)+1)

    return nothing
end

function update_vertical_velocities!(velocities, grid, model; parameters = w_kernel_parameters(grid))

    update_grid_vertical_velocity!(velocities, grid, model.vertical_coordinate; parameters)
    compute_w_from_continuity!(velocities, architecture(grid), grid; parameters)
    
    return nothing
end