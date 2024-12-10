using Oceananigans: UpdateStateCallsite
using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: update_boundary_condition!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.Fields: compute!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models: update_model_field_time_series!, surface_kernel_parameters, interior_tendency_kernel_parameters

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

    # Update all FieldTimeSeries used in the model
    update_model_field_time_series!(model, model.clock)

    # Update the boundary conditions
    update_boundary_condition!(fields(model), model)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.velocities, model.tracers), model.clock, fields(model); 
                       fill_boundary_normal_velocities = false, async = true)

    # Compute auxiliary fields
    for aux_field in model.auxiliary_fields
        compute!(aux_field)
    end

    # Calculate diffusivities and hydrostatic pressure
    @apply_regionally compute_auxiliaries!(model, grid, arch)
    fill_halo_regions!(model.diffusivity_fields; only_local_halos = true)
    
    for callback in callbacks
        callback.callsite isa UpdateStateCallsite && callback(model)
    end

    update_biogeochemical_state!(model.biogeochemistry, model)

    compute_tendencies && 
        @apply_regionally compute_tendencies!(model, callbacks)

    return nothing
end

function compute_auxiliaries!(model::NonhydrostaticModel, grid, arch; 
                              params2D = surface_kernel_parameters(arch, grid),
                              params3D = interior_tendency_kernel_parameters(arch, grid))

    closure = model.closure
    diffusivity = model.diffusivity_fields

    update_hydrostatic_pressure!(model; parameters=params2D)

    compute_diffusivities!(diffusivity, closure, model; parameters=params3D)

    return nothing
end
