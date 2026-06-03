using Oceananigans.BoundaryConditions: compute_x_bcs!, compute_y_bcs!, compute_z_bcs!
using Oceananigans.TimeSteppers: TimeSteppers

#####
##### Boundary contributions to hydrostatic free surface model
#####

function refresh_hydrostatic_flux_boundary_condition_state!(model::HydrostaticFreeSurfaceModel)
    update_model_field_time_series!(model, model.clock)

    @apply_regionally begin
        compute_auxiliary_fields!(model.auxiliary_fields)
        update_boundary_conditions!(fields(model), model)
        update_prescribed_velocity_field_operations!(model.velocities)
    end

    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model); async=false)
    @apply_regionally compute_w_from_continuity!(model)
    compute_auxiliary_fields!(model.auxiliary_fields)
    fill_halo_regions!(model.velocities, model.clock, fields(model); async=false)
    compute_auxiliary_fields!(model.auxiliary_fields)

    return nothing
end

function compute_flux_bcs!(Gcⁿ, c, arch, args)
    compute_x_bcs!(Gcⁿ, c, arch, args...)
    compute_y_bcs!(Gcⁿ, c, arch, args...)
    compute_z_bcs!(Gcⁿ, c, arch, args...)
    return nothing
end

@inline function compute_momentum_flux_bcs!(model::HydrostaticFreeSurfaceModel)
    refresh_hydrostatic_flux_boundary_condition_state!(model)

    Gⁿ   = model.timestepper.Gⁿ
    grid = model.grid
    arch = architecture(grid)
    args = (model.clock, fields(model), model.closure, model.buoyancy)

    compute_flux_bcs!(Gⁿ.u, model.velocities.u, arch, args)
    compute_flux_bcs!(Gⁿ.v, model.velocities.v, arch, args)

    return nothing
end

@inline function compute_tracer_flux_bcs!(model::HydrostaticFreeSurfaceModel)
    refresh_hydrostatic_flux_boundary_condition_state!(model)

    Gⁿ   = model.timestepper.Gⁿ
    grid = model.grid
    arch = architecture(grid)
    args = (model.clock, fields(model), model.closure, model.buoyancy)

    for i in propertynames(model.tracers)
        compute_flux_bcs!(Gⁿ[i], model.tracers[i], arch, args)
    end

    return nothing
end
