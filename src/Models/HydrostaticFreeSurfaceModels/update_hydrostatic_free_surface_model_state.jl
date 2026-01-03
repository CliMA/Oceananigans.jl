using Oceananigans: UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!, update_boundary_conditions!
using Oceananigans.BuoyancyFormulations: compute_buoyancy_gradients!
using Oceananigans.Fields: compute!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Utils: KernelParameters
using Oceananigans.Models: update_model_field_time_series!, surface_kernel_parameters, volume_kernel_parameters
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!

import Oceananigans.TimeSteppers: update_state!

compute_auxiliary_fields!(auxiliary_fields) = Tuple(compute!(a) for a in auxiliary_fields)

# Note: see single_column_model_mode.jl for a "reduced" version of update_state! for
# single column models.

"""
    update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[])

Update the model state to be consistent with the current prognostic fields.

This function performs the following steps:
1. Mask immersed boundary regions (set field values to zero in solid regions)
2. Update time-dependent field boundary conditions
3. Fill halo regions for velocities and tracers
4. Compute diagnostic quantities:
   - Buoyancy gradients (for turbulence closures)
   - Vertical velocity `w` from continuity equation
   - Hydrostatic pressure `pHY′`
   - Turbulent diffusivities
5. Fill local halos for closure fields and pressure
6. Execute any callbacks registered for `UpdateStateCallsite`
7. Update biogeochemical state

Note: Halo regions for free surface fields are filled separately after the barotropic step.
"""
update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[]) =  update_state!(model, model.grid, callbacks)

function update_state!(model::HydrostaticFreeSurfaceModel, grid, callbacks)

    arch = architecture(grid)

    @apply_regionally begin
        mask_immersed_model_fields!(model)
        update_model_field_time_series!(model, model.clock)
        update_boundary_conditions!(fields(model), model)
    end
    
    u = model.velocities.u
    v = model.velocities.v
    tracers = model.tracers

    # Fill the halos of the prognostic fields. Note that the halos of the 
    # free-surface variables are filled after the barotropic step.
    fill_halo_regions!((u, v),  model.clock, fields(model); async=true)
    fill_halo_regions!(tracers, model.clock, fields(model); async=true)

    # Compute diagnostic quantities
    @apply_regionally begin
        surface_params = surface_kernel_parameters(grid)
        volume_params = volume_kernel_parameters(grid)
        κ_params = diffusivity_kernel_parameters(grid)
        compute_buoyancy_gradients!(model.buoyancy, grid, tracers, parameters=volume_params)
        update_vertical_velocities!(model.velocities, grid, model, parameters=surface_params)    
        update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers, parameters=surface_params)
        compute_diffusivities!(model.closure_fields, model.closure, model, parameters=κ_params)
    end

    # Fill only local halos for diagnostic quantities since the parameters used
    # above include regions inside the (horizontal) halos.
    fill_halo_regions!(model.closure_fields; only_local_halos=true)
    fill_halo_regions!(model.pressure.pHY′; only_local_halos=true)

    [callback(model) for callback in callbacks if callback.callsite isa UpdateStateCallsite]

    update_biogeochemical_state!(model.biogeochemistry, model)

    @apply_regionally compute_momentum_tendencies!(model, callbacks)

    return nothing
end

"""
    mask_immersed_model_fields!(model)

Set field values to zero in immersed (solid) regions of the grid.

Masks both velocity fields and tracers to ensure physically meaningful values
at immersed boundaries. This is called at the beginning of `update_state!`.
"""
function mask_immersed_model_fields!(model)
    mask_immersed_velocities!(model.velocities)
    foreach(mask_immersed_field!, model.tracers)
    return nothing
end

"""
    mask_immersed_velocities!(velocities)

Set velocity field values to zero in immersed (solid) regions of the grid.
"""
mask_immersed_velocities!(velocities) = foreach(mask_immersed_field!, velocities)

"""
    diffusivity_kernel_parameters(grid)

Return kernel parameters for computing turbulent diffusivities including one extra cell
in horizontal directions.

The extra cells (indices `0:Nx+1` and `0:Ny+1`) are needed because diffusivities at
cell faces require data from neighboring cells. This ensures that viscous fluxes
can be computed correctly at domain boundaries without requiring (possibly costly) halo exchanges.
"""
@inline function diffusivity_kernel_parameters(grid)
    Nx, Ny, Nz = size(grid)
    Tx, Ty, Tz = topology(grid)

    ii = ifelse(Tx == Flat, 1:Nx, 0:Nx+1)
    jj = ifelse(Ty == Flat, 1:Ny, 0:Ny+1)
    kk = 1:Nz

    return KernelParameters(ii, jj, kk)
end
