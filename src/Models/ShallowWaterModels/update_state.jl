using Oceananigans.Architectures
using Oceananigans.BoundaryConditions

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model)

Update peripheral aspects of the model (halo regions, diffusivities, hydrostatic pressure) to the current model state.
"""
function update_state!(model::ShallowWaterModel)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.solution, model.tracers), model.architecture, 
                       model.clock, fields(model))

    return nothing
end

