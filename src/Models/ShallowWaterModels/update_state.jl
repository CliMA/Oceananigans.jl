using Oceananigans.Architectures
using Oceananigans.BoundaryConditions

#FJP: why is this here?
#import Oceananigans.TimeSteppers: update_state!

using Oceananigans: AbstractModel

"""
    update_state!(model)

Update peripheral aspects of the model (halo regions, diffusivities, hydrostatic pressure) to the current model state.

FJP: In ShallowWaterModels.

"""
function update_state!(model::AbstractModel)

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.solution, model.tracers), model.architecture, 
                       model.clock, fields(model))

    return nothing
end

