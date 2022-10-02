using Oceananigans.ImmersedBoundaries: mask_immersed_field!

import Oceananigans.TimeSteppers: update_state!
import Oceananigans.BoundaryConditions: fill_halo_regions!

"""
    update_state!(model::ShallowWaterModel)

Fill halo regions for `model.solution` and `model.tracers`.
"""
function update_state!(model::ShallowWaterModel)

    # Mask immersed fields
    masking_events = Tuple(mask_immersed_field!(field) for field in model.solution)

    wait(device(model.architecture), MultiEvent(masking_events))
    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)
    
    # Compute the velocities
    compute_velocities!(model.velocities, formulation(model))

    return nothing
end

compute_velocities!(U, ::VectorInvariantFormulation) = nothing

function compute_velocities!(U, ::ConservativeFormulation)
    compute!(U.u)
    compute!(U.v)
end

fill_halo_regions!(model::ShallowWaterModel; async=false) = fill_halo_regions!(merge(model.solution, model.tracers), model.clock, fields(model); async)