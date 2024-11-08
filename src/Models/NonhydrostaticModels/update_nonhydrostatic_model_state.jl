using Oceananigans: UpdateStateCallsite
using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: update_boundary_condition!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.Fields: compute!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models: update_model_field_time_series!

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
    @info "        masked immersed tracers"

    # Update all FieldTimeSeries used in the model
    update_model_field_time_series!(model, model.clock)
    @info "        updated time series"

    # Update the boundary conditions
    update_boundary_condition!(fields(model), model)
    @info "        updated boundary conditions"

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.velocities, model.tracers), model.clock, fields(model); 
                       fill_boundary_normal_velocities = false, async = true)
    @info "        filled halo regions"

    # Compute auxiliary fields
    for aux_field in model.auxiliary_fields
        compute!(aux_field)
    end
    @info "        computed aux fields"

    # Calculate diffusivities and hydrostatic pressure
    @apply_regionally compute_auxiliaries!(model)
    fill_halo_regions!(model.diffusivity_fields; only_local_halos = true)
    @info "        computed auxiliaries and filled regions"
    
    for callback in callbacks
        callback.callsite isa UpdateStateCallsite && callback(model)
    end

    update_biogeochemical_state!(model.biogeochemistry, model)

    compute_tendencies && 
        @apply_regionally compute_tendencies!(model, callbacks)
    @info "        finished updating"

    return nothing
end

function compute_auxiliaries!(model::NonhydrostaticModel; p_parameters = tuple(p_kernel_parameters(model.grid)),
                                                          κ_parameters = tuple(:xyz)) 

    closure = model.closure
    diffusivity = model.diffusivity_fields

    for (ppar, κpar) in zip(p_parameters, κ_parameters)
        compute_diffusivities!(diffusivity, closure, model; parameters = κpar)
        update_hydrostatic_pressure!(model; parameters = ppar)
    end
    return nothing
end
