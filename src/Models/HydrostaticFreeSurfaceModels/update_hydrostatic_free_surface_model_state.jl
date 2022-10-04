using Oceananigans.Architectures
using Oceananigans.Architectures: device_event
using Oceananigans.BoundaryConditions
using Oceananigans.TurbulenceClosures: calculate_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_reduced_field_xy!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!

import Oceananigans.TimeSteppers: update_state!
import Oceananigans.BoundaryConditions: fill_halo_regions!

compute_auxiliary_fields!(auxiliary_fields) = Tuple(compute!(a) for a in auxiliary_fields)

"""
    update_state!(model::HydrostaticFreeSurfaceModel)

Update peripheral aspects of the model (auxiliary fields, halo regions, diffusivities,
hydrostatic pressure) to the current model state.
"""
update_state!(model::HydrostaticFreeSurfaceModel) = update_state!(model, model.grid)

function update_state!(model::HydrostaticFreeSurfaceModel, grid)


    return nothing
end

function fill_halo_regions!(model::HydrostaticFreeSurfaceModel; async = false)

    η = displacement(model.free_surface)
    masking_events = Any[mask_immersed_field!(field)
                         for field in merge(model.auxiliary_fields, prognostic_fields(model)) if field !== η]
    push!(masking_events, mask_immersed_reduced_field_xy!(η, k=size(model.grid, 3)))    
    wait(device(model.architecture), MultiEvent(Tuple(masking_events)))

    calculate_diffusivities!(model.diffusivity_fields, model.closure, model)
    update_hydrostatic_pressure!(model.pressure.pHY′, model.architecture, model.grid, model.buoyancy, model.tracers)

    fill_horizontal_velocity_halos!(model.velocities.u, model.velocities.v, model.architecture)

    fill_halo_fields = merge(prognostic_fields(model), (pHY′ = model.pressure.pHY′, κ = model.diffusivity_fields))

    fill_halo_events = fill_halo_regions!(fill_halo_fields, model.clock, fields(model); async)

    interior_w_event = compute_w_from_continuity!(model; region_to_compute = :interior)

    wait(device(arch), interior_w_event)

    boundary_w_event = [compute_w_from_continuity!(model; region_to_compute = :west,  dependencies = fill_halo_events[end]), 
                        compute_w_from_continuity!(model; region_to_compute = :east,  dependencies = fill_halo_events[end]), 
                        compute_w_from_continuity!(model; region_to_compute = :south, dependencies = fill_halo_events[end]), 
                        compute_w_from_continuity!(model; region_to_compute = :north, dependencies = fill_halo_events[end])]


    fill_w_event = fill_halo_regions!(model.velocities.w, model.clock, fields(model); async, dependencies = boundary_w_event[end])

    events = Tuple(fill_halo_events..., boundary_w_event...)

    if !async
        wait(device(arch), events)
    end
    return events
end