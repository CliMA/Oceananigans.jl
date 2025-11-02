using Oceananigans.Fields: location, instantiated_location
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: get_active_cells_map, get_active_column_map

import Oceananigans.TimeSteppers: ssprk_substep!

ssprk_substep!(model::HydrostaticFreeSurfaceModel, Δt, callbacks) =
    ssprk_substep!(model, model.free_surface, model.grid, Δt, callbacks)

# SSPRK3 substep for hydrostatic free surface models, it differs in the order of operations
# depending on the type of free surface (implicit or explicit)
#
# For explicit free surfaces (`ExplicitFreeSurface` and `SplitExplicitFreeSurface`), we first
# compute the free surface using the integrated momentum baroclinic tendencies,
# then we advance grid, momentum and tracers. The last step is to reconcile the baroclinic and
# the barotropic modes by applying a pressure correction to momentum.
@inline function ssprk_substep!(model, free_surface, grid, Δt, callbacks)
    # Compute barotropic and baroclinic tendencies
    compute_momentum_tendencies!(model, callbacks)
    compute_free_surface_tendency!(grid, model, free_surface)

    # Advance the free surface first
    step_free_surface!(free_surface, model, model.timestepper, Δt)

    # Compute z-dependent transport velocities
    compute_transport_velocities!(model, free_surface)

    # compute tracer tendencies
    compute_tracer_tendencies!(model)

    if model.clock.stage == 2
        average_free_surface!(free_surface, model.timestepper)
    end

    # Advance grid and velocities
    ssprk_substep_grid!(grid, model, model.vertical_coordinate, Δt)
    ssprk_substep_velocities!(model.velocities, model, Δt)

    # Correct for the updated barotropic mode
    correct_barotropic_mode!(model, Δt)

    # TODO: fill halo regions for horizontal velocities should be here before the tracer update.   
    ssprk_substep_tracers!(model.tracers, model, Δt)

    return nothing
end

average_free_surface!(fs::ExplicitFreeSurface, ts) = 
    parent(fs.η) .= parent(fs.η) / 4 .+ 3 * parent(ts.Ψ⁻.η) / 4

function average_free_surface!(fs::SplitExplicitFreeSurface, ts) 
    U, V = fs.barotropic_velocities
    η    = fs.η

    parent(U) .= parent(U) ./ 4 .+ 3 .* parent(ts.Ψ⁻.U) ./ 4
    parent(V) .= parent(V) ./ 4 .+ 3 .* parent(ts.Ψ⁻.V) ./ 4
    parent(η) .= parent(η) ./ 4 .+ 3 .* parent(ts.Ψ⁻.η) ./ 4

    return nothing
end

# For implicit free surfaces (`ImplicitFreeSurface`), we first advance grid and tracers,
# we then use a predictor-corrector approach to advance momentum, in which we first
# advance momentum neglecting the free surface contribution, then, after the computation of
# the new free surface, we correct momentum to account for the updated free surface.
@inline ssprk_substep!(model, ::ImplicitFreeSurface, grid, Δt, callbacks) = throw(error("Not implemented!!"))

#####
##### Step grid
#####

# A Fallback to be extended for specific ztypes and grid types
ssprk_substep_grid!(grid, model, ztype::ZCoordinate, Δt) = nothing

#####
##### Step Velocities
#####

function ssprk_substep_velocities!(velocities, model, Δt)

    grid = model.grid
    FT = eltype(grid)

    θ = model.clock.stage == 2 ? 1/4 : 2/3 

    for name in (:u, :v)
        Gⁿ = model.timestepper.Gⁿ[name]
        Ψ⁻ = model.timestepper.Ψ⁻[name]
        velocity_field = velocities[name]

        launch!(architecture(grid), grid, :xyz,
                _euler_substep_field!, velocity_field, convert(FT, Δt), Gⁿ)

        implicit_step!(velocity_field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       nothing,
                       model.clock,
                       fields(model),
                       Δt)
        
        if model.clock.stage > 1
            launch!(architecture(grid), grid, :xyz, _ssp_average_field!, velocity_field, θ, Ψ⁻)
        end
    end

    return nothing
end

#####
##### Step Tracers
#####

ssprk_substep_tracers!(::EmptyNamedTuple, model, Δt) = nothing

function ssprk_substep_tracers!(tracers, model, Δt)

    closure = model.closure
    grid = model.grid
    FT = eltype(grid)

    catke_in_closures = hasclosure(closure, FlavorOfCATKE)

    θ = model.clock.stage == 2 ? 1/4 : 2/3 

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
                    _euler_substep_tracer_field!, c, grid, convert(FT, Δt), Gⁿ)

            implicit_step!(c,
                           model.timestepper.implicit_solver,
                           closure,
                           model.diffusivity_fields,
                           Val(tracer_index),
                           model.clock,
                           fields(model),
                           Δt)

            if model.clock.stage > 1
                launch!(architecture(grid), grid, :xyz, _ssp_average_tracer_field!, c, grid, θ, Ψ⁻)
            end
        end
    end

    return nothing
end

#####
##### Tracer update in mutable vertical coordinates
#####

# Velocity evolution kernel
@kernel function _euler_substep_field!(field, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] += Δt * Gⁿ[i, j, k]
end

# σc is the evolved quantity, so tracer fields need to be evolved
# accounting for the stretching factors from the new and the previous time step.
@kernel function _euler_substep_tracer_field!(c, grid, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    σᶜᶜ⁻ = σ⁻(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (σᶜᶜ⁻ * c[i, j, k] + Δt * Gⁿ[i, j, k]) / σᶜᶜⁿ
end

@kernel function _ssp_average_field!(U, θ, Ψ⁻) 
    i, j, k = @index(Global, NTuple)
    @inbounds U[i, j, k] = θ * U[i, j, k] + (1 - θ) * Ψ⁻[i, j, k]
end

@kernel function _ssp_average_tracer_field!(c, grid, θ, Ψ⁻) 
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = θ * c[i, j, k] + (1 - θ) * Ψ⁻[i, j, k] / σᶜᶜⁿ
end