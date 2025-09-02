import Oceananigans.TimeSteppers: compute_flux_bc_tendencies!

#####
##### Boundary contributions to hydrostatic free surface model
#####

function compute_flux_bcs!(Gcⁿ, c, arch, args)
    compute_x_bcs!(Gcⁿ, c, arch, args...)
    compute_y_bcs!(Gcⁿ, c, arch, args...)
    compute_z_bcs!(Gcⁿ, c, arch, args...)
    return nothing
end

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function compute_flux_bc_tendencies!(model::HydrostaticFreeSurfaceModel)
    compute_tracers_flux_bcs!(model)
    compute_momentum_flux_bcs!(model)
    return nothing
end

@inline function compute_momentum_flux_bcs!(model::HydrostaticFreeSurfaceModel) 
    Gⁿ   = model.timestepper.Gⁿ
    grid = model.grid
    arch = architecture(grid)
    args = (model.clock, fields(model), model.closure, model.buoyancy)

    @apply_regionally compute_flux_bcs!(Gⁿ.u, model.velocities.u, arch, args)
    @apply_regionally compute_flux_bcs!(Gⁿ.v, model.velocities.v, arch, args)

    return nothing
end

@inline function compute_tracers_flux_bcs!(model::HydrostaticFreeSurfaceModel) 
    Gⁿ   = model.timestepper.Gⁿ
    grid = model.grid
    arch = architecture(grid)
    args = (model.clock, fields(model), model.closure, model.buoyancy)

    for i in propertynames(model.tracers)
        @apply_regionally compute_flux_bcs!(Gⁿ[i], model.tracers[i], arch, args)
    end

    return nothing
end
