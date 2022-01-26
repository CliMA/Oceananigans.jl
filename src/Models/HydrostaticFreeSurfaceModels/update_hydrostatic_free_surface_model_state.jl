using Oceananigans.Architectures
using Oceananigans.Architectures: device_event
using Oceananigans.BoundaryConditions
using Oceananigans.TurbulenceClosures: calculate_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_reduced_field_xy!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!

import Oceananigans.TimeSteppers: update_state!

compute_auxiliary_fields!(auxiliary_fields) = Tuple(compute!(a) for a in auxiliary_fields)

"""
    update_state!(model::HydrostaticFreeSurfaceModel)

Update peripheral aspects of the model (auxiliary fields, halo regions, diffusivities,
hydrostatic pressure) to the current model state.
"""
update_state!(model::HydrostaticFreeSurfaceModel) = update_state!(model, model.grid)

function update_state!(model::HydrostaticFreeSurfaceModel, grid)

    # Mask immersed fields
    η = displacement(model.free_surface)

    masking_events = Any[mask_immersed_field!(field)
                         for field in merge(model.auxiliary_fields, prognostic_fields(model)) if field !== η]

    push!(masking_events, mask_immersed_reduced_field_xy!(η, k=size(grid, 3)))

    wait(device(model.architecture), MultiEvent(Tuple(masking_events)))

    # Fill halos for velocities and tracers. On the CubedSphere, the halo filling for velocity fields is wrong.
    fill_halo_regions!(prognostic_fields(model), model.architecture, model.clock, fields(model))

    # This _refills_ the halos for horizontal velocity fields when grid::ConformalCubedSphereGrid
    # For every other type of grid, fill_horizontal_velocity_halos! does nothing.
    fill_horizontal_velocity_halos!(model.velocities.u, model.velocities.v, model.architecture)

    compute_w_from_continuity!(model)

    fill_halo_regions!(model.velocities.w, model.architecture, model.clock, fields(model))

    compute_auxiliary_fields!(model.auxiliary_fields)

    # Calculate diffusivities
    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)
    fill_halo_regions!(model.diffusivity_fields, model.architecture, model.clock, fields(model))

    update_hydrostatic_pressure!(model.pressure.pHY′, model.architecture, model.grid, model.buoyancy, model.tracers)
    fill_halo_regions!(model.pressure.pHY′, model.architecture)

    return nothing
end
