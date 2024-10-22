using Oceananigans.Architectures
using Oceananigans.BoundaryConditions

using Oceananigans: UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: update_boundary_condition!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!, inactive_node
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!, p_kernel_parameters

import Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!
import Oceananigans.TimeSteppers: update_state!

compute_auxiliary_fields!(auxiliary_fields) = Tuple(compute!(a) for a in auxiliary_fields)

# Note: see single_column_model_mode.jl for a "reduced" version of update_state! for
# single column models.

"""
    update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[]; compute_tendencies = true)

Update peripheral aspects of the model (auxiliary fields, halo regions, diffusivities,
hydrostatic pressure) to the current model state. If `callbacks` are provided (in an array),
they are called in the end. Finally, the tendencies for the new time-step are computed if 
`compute_tendencies = true`.
"""
update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[]; compute_tendencies = true) =
    update_state!(model, model.grid, callbacks; compute_tendencies)

function update_state!(model::HydrostaticFreeSurfaceModel, grid, callbacks; compute_tendencies = true)
    @apply_regionally mask_immersed_model_fields!(model, grid)

    # Update possible FieldTimeSeries used in the model
    @apply_regionally update_model_field_time_series!(model, model.clock)

    # Update the boundary conditions
    @apply_regionally update_boundary_condition!(fields(model), model)

    if grid isa ConformalCubedSphereGrid
        if model.tracers != nothing
            for tracer in model.tracers
                fill_halo_regions!(tracer)
            end
        end
        if model.free_surface != nothing
            fill_halo_regions!(model.free_surface.η)
        end
        fill_halo_regions!((model.velocities.u, model.velocities.v))
    else
        fill_halo_regions!(prognostic_fields(model), model.clock, fields(model); async = true)
    end

    @apply_regionally compute_auxiliaries!(model)

    fill_halo_regions!(model.diffusivity_fields; only_local_halos = true)

    [callback(model) for callback in callbacks if callback.callsite isa UpdateStateCallsite]

    update_biogeochemical_state!(model.biogeochemistry, model)

    compute_tendencies &&
        @apply_regionally compute_tendencies!(model, callbacks)

    return nothing
end

# Mask immersed fields
function mask_immersed_model_fields!(model, grid)
    η = displacement(model.free_surface)
    fields_to_mask = merge(model.auxiliary_fields, prognostic_fields(model))

    foreach(fields_to_mask) do field
        if field !== η
            mask_immersed_field!(field)
        end
    end
    mask_immersed_field_xy!(η, k=size(grid, 3)+1, mask = inactive_node)

    return nothing
end

function compute_auxiliaries!(model::HydrostaticFreeSurfaceModel; w_parameters = tuple(w_kernel_parameters(model.grid)),
                                                                  p_parameters = tuple(p_kernel_parameters(model.grid)),
                                                                  κ_parameters = tuple(:xyz))

    grid = model.grid
    closure = model.closure
    diffusivity = model.diffusivity_fields

    for (wpar, ppar, κpar) in zip(w_parameters, p_parameters, κ_parameters)
        compute_w_from_continuity!(model; parameters = wpar)
        compute_diffusivities!(diffusivity, closure, model; parameters = κpar)
        update_hydrostatic_pressure!(model.pressure.pHY′, architecture(grid),
                                     grid, model.buoyancy, model.tracers; 
                                     parameters = ppar)
    end
    return nothing
end
