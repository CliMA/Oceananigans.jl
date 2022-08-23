using Oceananigans.BoundaryConditions: fill_halo_regions!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::ShallowWaterModel)

Fill halo regions for `model.solution` and `model.tracers`.
"""
function update_state!(model::ShallowWaterModel)

    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.solution, model.tracers),
                       model.clock,
                       fields(model))

    # Compute the velocities

    compute_velocities!(model.velocities, formulation(model))

    return nothing
end

compute_velocities!(U, ::VectorInvariantFormulation) = nothing

function compute_velocities!(U, ::ConservativeFormulation)
    compute!(U.u)
    compute!(U.v)
end
