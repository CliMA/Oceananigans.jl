using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::ShallowWaterModel, callbacks=[])

Fill halo regions for `model.solution` and `model.tracers`.
If `callbacks` are provided (in an array), they are called in the end.
"""
function update_state!(model::ShallowWaterModel, callbacks=[])

    # Mask immersed fields
    masking_events = Tuple(mask_immersed_field!(field) for field in model.solution)

    wait(device(model.architecture), MultiEvent(masking_events))

    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.solution, model.tracers),
                       model.clock,
                       fields(model))

    # Compute the velocities

    compute_velocities!(model.velocities, formulation(model))

    [callback(model) for callback in callbacks if isa(callback.callsite, UpdateStateCallsite)]

    return nothing
end

compute_velocities!(U, ::VectorInvariantFormulation) = nothing

function compute_velocities!(U, ::ConservativeFormulation)
    compute!(U.u)
    compute!(U.v)
end
