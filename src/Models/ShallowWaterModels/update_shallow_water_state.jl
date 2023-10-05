using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models: update_model_field_time_series!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::ShallowWaterModel, callbacks=[])

Fill halo regions for `model.solution` and `model.tracers`.
If `callbacks` are provided (in an array), they are called in the end.
"""
function update_state!(model::ShallowWaterModel, callbacks=[]; compute_tendencies = true)

    # Mask immersed fields
    foreach(mask_immersed_field!, model.solution)

    # Update possible FieldTimeSeries used in the model
    update_model_field_time_series!(model, model.clock)
    
    compute_diffusivities!(model.diffusivity_fields, model.closure, model)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.solution, model.tracers), model.clock, fields(model))

    # Compute the velocities

    compute_velocities!(model.velocities, formulation(model))

    foreach(callbacks) do callback
        if isa(callback.callsite, UpdateStateCallsite)
            callback(model)
        end
    end

    compute_tendencies && compute_tendencies!(model, callbacks)

    return nothing
end

compute_velocities!(U, ::VectorInvariantFormulation) = nothing

function compute_velocities!(U, ::ConservativeFormulation)
    compute!(U.u)
    compute!(U.v)
    return nothing
end
