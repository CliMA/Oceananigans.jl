using Oceananigans.Fields: location, instantiated_location
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: get_active_cells_map, get_active_column_map

import Oceananigans.TimeSteppers: rk3_substep!, cache_previous_fields!

rk3_substep!(model::HydrostaticFreeSurfaceModel, grid, Δτ, callbacks) = 
    rk3_substep!(model, model.free_surface, grid, Δτ, callbacks)

# RK3 substep for hydrostatic free surface models, it differs in the order of operations
# depending on the type of free surface (implicit or explicit)
#
# For explicit free surfaces (`ExplicitFreeSurface` and `SplitExplicitFreeSurface`), we first
# compute the free surface that depends on the momentum baroclinic tendencies,
# then we advance grid, momentum and tracers.
# 
# For implicit free surfaces (`ImplicitFreeSurface`), we first advance grid and tracers,
# we then use a predictor-corrector approach to advance momentum, in which we first
# advance momentum neglecting the free surface contribution, then, after the computation of
# the new free surface, we correct momentum to account for the updated free surface.
@inline function rk3_substep!(model, free_surface, grid, Δτ, callbacks)

    # Advancing free surface and barotropic transport velocities
    compute_momentum_tendencies!(model, callbacks)
    compute_free_surface_tendency!(grid, model, model.free_surface)
    step_free_surface!(model.free_surface, model, model.timestepper, Δτ)
    
    # Computing z-dependent transport velocities
    compute_transport_velocities!(model, model.free_surface)
    
    # compute tracer tendencies
    compute_tracer_tendencies!(model)
    
    # Remember to scale tracers tendencies by stretching factor
    scale_by_stretching_factor!(model.timestepper.Gⁿ, model.tracers, model.grid)
    
    # Finally Substep! Advance grid, tracers, and momentum
    rk3_substep_grid!(grid, model, model.vertical_coordinate, Δτ)
    rk3_substep_tracers!(model.tracers, model, Δτ)
    rk3_substep_velocities!(model.velocities, model, Δτ)

    # Correct for the updated barotropic mode
    make_pressure_correction!(model, Δτ)

    return nothing
end

@inline function rk3_substep!(model, ::ImplicitFreeSurface, grid, Δτ, callbacks)

    # Computing tendencies...
    compute_momentum_tendencies!(model, callbacks)
    compute_tracer_tendencies!(model)
    compute_free_surface_tendency!(grid, model, model.free_surface)
    
    # Remember to scale tracers tendencies by stretching factor
    scale_by_stretching_factor!(model.timestepper.Gⁿ, model.tracers, model.grid)
    
    # Finally Substep! Advance grid, tracers, momentum and free surface
    rk3_substep_grid!(grid, model, model.vertical_coordinate, Δτ)
    rk3_substep_tracers!(model.tracers, model, Δτ)
    rk3_substep_velocities!(model.velocities, model, Δτ)
    step_free_surface!(model.free_surface, model, model.timestepper, Δτ)

    # Correct for the updated barotropic mode
    make_pressure_correction!(model, Δτ)

    return nothing
end

#####
##### Step Velocities
#####

function rk3_substep_velocities!(velocities, model, Δt)

    grid = model.grid
    FT = eltype(grid)

    for name in (:u, :v)
        Gⁿ = model.timestepper.Gⁿ[name]
        Ψ⁻ = model.timestepper.Ψ⁻[name]
        velocity_field = velocities[name]

        launch!(architecture(grid), grid, :xyz,
                _euler_substep_field!, velocity_field, convert(FT, Δt), Gⁿ, Ψ⁻)

        implicit_step!(velocity_field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       nothing,
                       model.clock,
                       Δt)
    end

    return nothing
end

#####
##### Step Tracers
#####

rk3_substep_tracers!(::EmptyNamedTuple, model, Δt) = nothing

function rk3_substep_tracers!(tracers, model, Δt)

    closure = model.closure
    grid = model.grid
    FT = eltype(grid)

    catke_in_closures = hasclosure(closure, FlavorOfCATKE)

    # Tracer update kernels
    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))

        if catke_in_closures && tracer_name == :e
            @debug "Skipping RK3 step for e"
        else
            Gⁿ = model.timestepper.Gⁿ[tracer_name]
            Ψ⁻ = model.timestepper.Ψ⁻[tracer_name]
            c  = tracers[tracer_name]
            closure = model.closure

            launch!(architecture(grid), grid, :xyz,
                    _euler_substep_tracer_field!, c, grid, convert(FT, Δt), Gⁿ, Ψ⁻)

            implicit_step!(c,
                           model.timestepper.implicit_solver,
                           closure,
                           model.diffusivity_fields,
                           Val(tracer_index),
                           model.clock,
                           Δt)
        end
    end

    return nothing
end

#####
##### Tracer update in mutable vertical coordinates
#####

# Velocity evolution kernel
@kernel function _euler_substep_field!(field, Δt, Gⁿ, Ψ⁻)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = Ψ⁻[i, j, k] + Δt * Gⁿ[i, j, k]
end

# σc is the evolved quantity, so tracer fields need to be evolved
# accounting for the stretching factors from the new and the previous time step.
@kernel function _euler_substep_tracer_field!(c, grid, Δt, Gⁿ, σc⁻)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (σc⁻[i, j, k] + Δt * Gⁿ[i, j, k]) / σᶜᶜⁿ
end

#####
##### Storing previous fields for the RK3 update
#####

# Tracers are multiplied by the vertical coordinate scaling factor
@kernel function _cache_tracer_fields!(Ψ⁻, grid, Ψⁿ)
    i, j, k = @index(Global, NTuple)
    @inbounds Ψ⁻[i, j, k] = Ψⁿ[i, j, k] * σⁿ(i, j, k, grid, Center(), Center(), Center())
end

function cache_previous_fields!(model::HydrostaticFreeSurfaceModel)

    previous_fields = model.timestepper.Ψ⁻
    model_fields = prognostic_fields(model)
    grid = model.grid
    arch = architecture(grid)

    for name in keys(model_fields)
        Ψ⁻ = previous_fields[name]
        Ψⁿ = model_fields[name]
        if name ∈ keys(model.tracers) # Tracers are stored with the grid scaling
            launch!(arch, grid, :xyz, _cache_tracer_fields!, Ψ⁻, grid, Ψⁿ)
        else # Velocities and free surface are stored without the grid scaling
            parent(Ψ⁻) .= parent(Ψⁿ)
        end
    end

    cache_grid_state!(model.vertical_coordinate, grid, model.free_surface)

    return nothing
end

cache_grid_state!(ztype, grid, free_surface) = nothing
cache_grid_state!(ztype::ZStarCoordinate, grid, ::ImplicitFreeSurface) = parent(ztype.storage) .= parent(grid.z.ηⁿ)
