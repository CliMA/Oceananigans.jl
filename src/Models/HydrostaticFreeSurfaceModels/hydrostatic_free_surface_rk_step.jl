using Oceananigans.Fields: location, instantiated_location
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: get_active_cells_map, get_active_column_map

import Oceananigans.TimeSteppers: rk_substep!, cache_previous_fields!

rk_substep!(model::HydrostaticFreeSurfaceModel, Δτ, callbacks) =
    rk_substep!(model, model.free_surface, model.grid, Δτ, callbacks)

# RK3 substep for hydrostatic free surface models, it differs in the order of operations
# depending on the type of free surface (implicit or explicit)
#
# For explicit free surfaces (`ExplicitFreeSurface` and `SplitExplicitFreeSurface`), we first
# compute the free surface using the integrated momentum baroclinic tendencies,
# then we advance grid, momentum and tracers. The last step is to reconcile the baroclinic and
# the barotropic modes by applying a pressure correction to momentum.
@inline function rk_substep!(model, free_surface, grid, Δτ, callbacks)
    # Compute barotropic and baroclinic tendencies
    @apply_regionally compute_momentum_tendencies!(model, callbacks)
    
    # Advance the free surface first
    compute_free_surface_tendency!(grid, model, free_surface)
    step_free_surface!(free_surface, model, model.timestepper, Δτ)

    # Compute z-dependent transport velocities
    compute_transport_velocities!(model, free_surface)

    @apply_regionally begin
        # Advance grid and velocities
        rk_substep_velocities!(model.velocities, model, Δτ)
        rk_substep_grid!(grid, model, model.vertical_coordinate, Δτ)

        # Correct for the updated barotropic mode
        correct_barotropic_mode!(model, Δτ)
    end
    
    update_velocity_state!(model.velocities, model)

    # Compute tracer tendencies and advance tracers
    @apply_regionally begin
        compute_tracer_tendencies!(model)
        rk_substep_tracers!(model.tracers, model, Δτ)
    end
    
    return nothing
end

# For implicit free surfaces (`ImplicitFreeSurface`), we first advance grid and tracers,
# we then use a predictor-corrector approach to advance momentum, in which we first
# advance momentum neglecting the free surface contribution, then, after the computation of
# the new free surface, we correct momentum to account for the updated free surface.
@inline function rk_substep!(model, ::ImplicitFreeSurface, grid, Δτ, callbacks)

    @apply_regionally begin
        # Computing tendencies...
        compute_momentum_tendencies!(model, callbacks)
        compute_tracer_tendencies!(model)
        
        # Finally Substep! Advance grid, tracers, (predictor) momentum 
        rk_substep_grid!(grid, model, model.vertical_coordinate, Δτ)
        rk_substep_velocities!(model.velocities, model, Δτ)
    end

    # Advancing free surface in preparation for the correction step
    step_free_surface!(model.free_surface, model, model.timestepper, Δτ)

    # Correct for the updated barotropic mode
    @apply_regionally correct_barotropic_mode!(model, Δτ)

    # TODO: fill halo regions for horizontal velocities should be here before the tracer update.   
    @apply_regionally rk_substep_tracers!(model.tracers, model, Δτ)

    update_velocity_state!(model.velocities, model)

    return nothing
end

#####
##### Step grid
#####

# A Fallback to be extended for specific ztypes and grid types
rk_substep_grid!(grid, model, ::ZCoordinate, Δt) = synchronize_communication!(model.free_surface)

#####
##### Step Velocities
#####

function rk_substep_velocities!(velocities, model, Δt)

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
                       model.closure_fields,
                       nothing,
                       model.clock,
                       fields(model),
                       Δt)
    end

    return nothing
end

#####
##### Step Tracers
#####

rk_substep_tracers!(::EmptyNamedTuple, model, Δt) = nothing

function rk_substep_tracers!(tracers, model, Δt)

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
                           model.closure_fields,
                           Val(tracer_index),
                           model.clock,
                           fields(model),
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
