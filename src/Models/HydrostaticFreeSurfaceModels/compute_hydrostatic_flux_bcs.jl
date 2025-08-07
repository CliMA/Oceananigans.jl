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

    Gⁿ         = model.timestepper.Gⁿ
    grid       = model.grid
    arch       = architecture(grid)
    velocities = model.velocities
    tracers    = model.tracers
    
    args = (model.clock, fields(model), model.closure, model.buoyancy)

    
    # Velocity fields
    for i in (:u, :v)
        @apply_regionally compute_flux_bcs!(Gⁿ[i], velocities[i], arch, args)
    end

    # Tracer fields
    for i in propertynames(tracers)
        @apply_regionally compute_flux_bcs!(Gⁿ[i], tracers[i], arch, args)
    end

    return nothing
end
