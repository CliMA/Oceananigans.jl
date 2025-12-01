using Oceananigans: UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!, update_boundary_conditions!
using Oceananigans.BuoyancyFormulations: compute_buoyancy_gradients!
using Oceananigans.Fields: compute!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!, surface_kernel_parameters

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
        mask_immersed_model_fields!(model)
        update_model_field_time_series!(model, model.clock)
        update_boundary_conditions!(fields(model), model)
    end
    
    u = model.velocities.u
    v = model.velocities.v
    tracers = model.tracers

    # Fill the halos
    fill_halo_regions!((u, v),  grid, model.clock, fields(model); async=true)
    fill_halo_regions!(tracers, grid, model.clock, fields(model); async=true)

    @apply_regionally begin
        surface_params = surface_kernel_parameters(model.grid)
        compute_buoyancy_gradients!(model.buoyancy, grid, tracers, parameters=:xyz)
        update_vertical_velocities!(model.velocities, model.grid, model, parameters=surface_params)    
        update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers, parameters=surface_params)
        compute_diffusivities!(model.closure_fields, model.closure, model, parameters=:xyz)
    end

    fill_halo_regions!(model.closure_fields; only_local_halos=true)
    fill_halo_regions!(model.pressure.pHY′;  only_local_halos=true)

    [callback(model) for callback in callbacks if callback.callsite isa UpdateStateCallsite]

    update_biogeochemical_state!(model.biogeochemistry, model)

    return nothing
end

# Mask immersed fields
function mask_immersed_model_fields!(model)
    mask_immersed_velocities!(model.velocities)
    foreach(mask_immersed_field!, model.tracers)
    return nothing
end

mask_immersed_velocities!(velocities) = foreach(mask_immersed_field!, velocities)
