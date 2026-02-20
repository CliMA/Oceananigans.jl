using Oceananigans: UpdateStateCallsite
using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: update_boundary_conditions!
using Oceananigans.BuoyancyFormulations: compute_buoyancy_gradients!
using Oceananigans.TurbulenceClosures: compute_closure_fields!
import Oceananigans.TurbulenceClosures: step_closure_prognostics!
using Oceananigans.Fields: compute!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models: update_model_field_time_series!, surface_kernel_parameters

"""
    update_state!(model::NonhydrostaticModel, callbacks=[])

Update peripheral aspects of the model (halo regions, closure_fields, hydrostatic
pressure) to the current model state. If `callbacks` are provided (in an array),
they are called in the end.
"""
function update_state!(model::NonhydrostaticModel, callbacks=[])

    # Mask immersed tracers
    foreach(model.tracers) do tracer
        mask_immersed_field!(tracer)
    end

    # Update all FieldTimeSeries used in the model
    update_model_field_time_series!(model, model.clock)

    # Update the boundary conditions
    update_boundary_conditions!(fields(model), model)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.velocities, model.tracers), model.clock, fields(model); fill_open_bcs=false, async=true)

    # Compute auxiliary fields
    for aux_field in model.auxiliary_fields
        compute!(aux_field)
    end

    # Calculate closure_fields and hydrostatic pressure
    compute_auxiliaries!(model)

    fill_halo_regions!(model.closure_fields; only_local_halos=true)
    fill_halo_regions!(model.pressures.pHY′; only_local_halos=true)

    for callback in callbacks
        callback.callsite isa UpdateStateCallsite && callback(model)
    end

    compute_tendencies!(model, callbacks)
    update_biogeochemical_state!(model.biogeochemistry, model)

    return nothing
end

function compute_auxiliaries!(model::NonhydrostaticModel; p_parameters = surface_kernel_parameters(model.grid),
                                                          κ_parameters = :xyz)

    grid = model.grid
    closure = model.closure
    closure_fields = model.closure_fields
    tracers = model.tracers
    buoyancy = model.buoyancy

    # Maybe compute buoyancy gradients
    compute_buoyancy_gradients!(buoyancy, grid, tracers; parameters = κ_parameters)

    # Compute closure_fields
    compute_closure_fields!(closure_fields, closure, model; parameters = κ_parameters)

    # Update hydrostatic pressure
    update_hydrostatic_pressure!(model; parameters = p_parameters)

    return nothing
end

step_closure_prognostics!(model::NonhydrostaticModel, Δt) =
    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)
