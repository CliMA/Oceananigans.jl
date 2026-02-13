using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: peripheral_node, MutableGridOfSomeKind

import Oceananigans.TimeSteppers: rk_substep!, cache_current_fields!

"""
    rk_substep!(model::HydrostaticFreeSurfaceModel, Δτ, callbacks)

Perform a single split Runge-Kutta substep for `HydrostaticFreeSurfaceModel`.

Dispatches to the appropriate method based on the free surface type (explicit or implicit).
The substep advances the state from the cached initial fields `Ψ⁻` using: `U = Ψ⁻ + Δτ * Gⁿ`.
"""
rk_substep!(model::HydrostaticFreeSurfaceModel, Δτ, callbacks) =
    rk_substep!(model, model.free_surface, model.grid, Δτ, callbacks)

"""
    rk_substep!(model, free_surface, grid, Δτ, callbacks)

Split Runge-Kutta substep for `HydrostaticFreeSurfaceModel` with explicit free surfaces
(`ExplicitFreeSurface` or `SplitExplicitFreeSurface`).

The order of operations for explicit free surfaces is:
1. Compute momentum tendencies (baroclinic)
2. Advance the free surface (barotropic step)
3. Compute transport velocities for tracer advection
4. Compute tracer tendencies
5. Advance grid scaling (for z-star coordinates)
6. Advance velocities
7. Correct barotropic mode to reconcile baroclinic and barotropic velocities
8. Advance tracers
"""
@inline function rk_substep!(model, free_surface, grid, Δτ, callbacks)
    # Compute barotropic and baroclinic tendencies
    @apply_regionally compute_momentum_flux_bcs!(model)

    # Advance the free surface first
    compute_free_surface_tendency!(grid, model, free_surface)
    step_free_surface!(free_surface, model, model.timestepper, Δτ)

    # Compute z-dependent transport velocities
    compute_transport_velocities!(model, free_surface)

    # correct_velocity_tendency!(model)

    @apply_regionally begin
        # compute tracer tendencies
        compute_tracer_tendencies!(model)

        # Advance grid and velocities
        rk_substep_grid!(grid, model, model.vertical_coordinate, Δτ)
        rk_substep_velocities!(model.velocities, model, Δτ)

        # Correct for the updated barotropic mode
        correct_barotropic_mode!(model, Δτ)

        # TODO: fill halo regions for horizontal velocities should be here before the tracer update.
        rk_substep_tracers!(model.tracers, model, Δτ)
    end

    return nothing
end

@kernel function _correct_velocity_tendency!(Gu, Gv, grid, advection, U, Ũ)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] += Oceananigans.Advection.div_𝐯u(i, j, k, grid, advection, U, U.u) -
                             Oceananigans.Advection.div_𝐯u(i, j, k, grid, advection, Ũ, U.u) 
    @inbounds Gu[i, j, k] += Oceananigans.Advection.div_𝐯v(i, j, k, grid, advection, U, U.v) -
                             Oceananigans.Advection.div_𝐯v(i, j, k, grid, advection, Ũ, U.v) 
end

function correct_velocity_tendency!(model)
    Gu = model.timestepper.Gⁿ.u
    Gv = model.timestepper.Gⁿ.v
    launch!(Oceananigans.CPU(), model.grid, :xyz, _correct_velocity_tendency!, 
            Gu, Gv, model.grid, model.advection.momentum, model.velocities, model.transport_velocities)
end

"""
    rk_substep!(model, ::ImplicitFreeSurface, grid, Δτ, callbacks)

Split Runge-Kutta substep for `HydrostaticFreeSurfaceModel` with `ImplicitFreeSurface`.

For implicit free surfaces, a predictor-corrector approach is used:
1. Compute momentum and tracer tendencies
2. Advance grid scaling (for z-star coordinates)
3. Advance velocities (predictor step, ignoring free surface)
4. Solve implicit free surface equation
5. Correct velocities for the updated barotropic pressure gradient
6. Advance tracers
"""
@inline function rk_substep!(model, free_surface::ImplicitFreeSurface, grid, Δτ, callbacks)

    @apply_regionally begin
        parent(model.transport_velocities.u) .= parent(model.velocities.u)
        parent(model.transport_velocities.v) .= parent(model.velocities.v)

        # Computing tendencies...
        compute_momentum_flux_bcs!(model)

        # Finally Substep! Advance grid, tracers, (predictor) momentum
        rk_substep_velocities!(model.velocities, model, Δτ)
    end

    # Advancing free surface in preparation for the correction step
    step_free_surface!(free_surface, model, model.timestepper, Δτ)

    # Correct for the updated barotropic mode
    @apply_regionally correct_barotropic_mode!(model, Δτ)

    compute_transport_velocities!(model, free_surface)

    @apply_regionally begin
        compute_tracer_tendencies!(model)

        rk_substep_grid!(model.grid, model, model.vertical_coordinate, Δτ)
        rk_substep_tracers!(model.tracers, model, Δτ)
    end

    return nothing
end

#####
##### Step grid
#####

"""
    rk_substep_grid!(grid, model, ::ZCoordinate, Δτ)

Update grid scaling factors during a split Runge-Kutta substep.

Fallback method that does nothing. Extended for `ZStarCoordinate` to update
the vertical grid spacing based on the new free surface height.
"""
rk_substep_grid!(grid, model, ::ZCoordinate, Δt) = nothing

#####
##### Step Velocities
#####

"""
    rk_substep_velocities!(velocities, model, Δτ)

Advance horizontal velocities `u` and `v` during a split Runge-Kutta substep.

Velocities are updated as: `uⁿ⁺¹ = u⁰ + Δτ * Gᵤ` where `u⁰` is the cached initial state
stored in `model.timestepper.Ψ⁻` and `Gᵤ` is the current tendency in `model.timestepper.Gⁿ`.

If an implicit solver is configured, implicit vertical diffusion is applied after the explicit step.
"""
function rk_substep_velocities!(velocities, model, Δt)

    grid = model.grid
    FT = eltype(grid)

    for name in (:u, :v)
        Gⁿ = model.timestepper.Gⁿ[name]
        Ψ⁻ = model.timestepper.Ψ⁻[name]
        velocity_field = velocities[name]

        launch!(architecture(grid), grid, :xyz,
                _rk_substep_field!, velocity_field, convert(FT, Δt), Gⁿ, Ψ⁻)

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

"""
    rk_substep_tracers!(tracers, model, Δτ)

Advance tracer fields during a split Runge-Kutta substep.

For mutable vertical coordinates (z-star), tracers are evolved accounting for the
grid stretching factor `σ`: the cached quantity is `σ⁰ * c⁰` and the update is
`c = (σ⁰ * c⁰ + Δτ * Gᶜ) / σⁿ` where `σⁿ` is the current stretching factor.

If CATKE closure is active, the TKE tracer `e` is skipped (handled separately).
Implicit vertical diffusion is applied after the explicit step if configured.
"""
function rk_substep_tracers!(tracers, model, Δt)

    closure = model.closure
    grid = model.grid
    FT = eltype(grid)

    catke_in_closures = hasclosure(closure, FlavorOfCATKE)

    # Tracer update kernels
    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))

        if catke_in_closures && tracer_name == :e
            @debug "Skipping RK substep for e"
        else
            Gⁿ = model.timestepper.Gⁿ[tracer_name]
            Ψ⁻ = model.timestepper.Ψ⁻[tracer_name]
            c  = tracers[tracer_name]

            launch!(architecture(grid), grid, :xyz,
                    _rk_substep_tracer_field!, c, grid, convert(FT, Δt), Gⁿ, Ψ⁻)

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
@kernel function _rk_substep_field!(field, Δt, Gⁿ, Ψ⁻)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = Ψ⁻[i, j, k] + Δt * Gⁿ[i, j, k]
end

# σc is the evolved quantity, so tracer fields need to be evolved
# accounting for the stretching factors from the new and the previous time step.
@kernel function _rk_substep_tracer_field!(c, grid, Δt, Gⁿ, σc⁻)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (σc⁻[i, j, k] + Δt * Gⁿ[i, j, k]) / σᶜᶜⁿ
end
