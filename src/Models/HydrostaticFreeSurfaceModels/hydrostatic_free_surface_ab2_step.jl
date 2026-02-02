using Oceananigans.TimeSteppers: _ab2_step_field!
using Oceananigans.Operators: σ⁻, σⁿ, ∂t_σ
using Oceananigans.TurbulenceClosures: implicit_step!

import Oceananigans.TimeSteppers: ab2_step!

#####
##### Step everything
#####

"""
    ab2_step!(model::HydrostaticFreeSurfaceModel, Δt, callbacks)

Advance `HydrostaticFreeSurfaceModel` by one Adams-Bashforth 2nd-order time step.

Dispatches to the appropriate method based on the free surface type (explicit or implicit).
"""
ab2_step!(model::HydrostaticFreeSurfaceModel, Δt, callbacks) =
    hydrostatic_ab2_step!(model, model.free_surface, model.grid, Δt, callbacks)

"""
    hydrostatic_ab2_step!(model, free_surface, grid, Δt, callbacks)

The Adams-Bashforth 2nd-order time step for `HydrostaticFreeSurfaceModel` with explicit free surfaces
(`ExplicitFreeSurface` or `SplitExplicitFreeSurface`).

The order of operations for explicit free surfaces is:
1. Compute momentum flux boundary conditions (3D tendencies are computed in `update_state!`)
2. Advance the free surface (barotropic step)
3. Compute transport velocities for tracer advection
4. Compute tracer tendencies
5. Advance grid scaling (for z-star coordinates)
6. Advance velocities using AB2
7. Correct barotropic mode
8. Advance tracers using AB2
"""
function hydrostatic_ab2_step!(model, free_surface, grid, Δt, callbacks)
    FT = eltype(grid)
    χ  = convert(FT, model.timestepper.χ)
    Δt = convert(FT, Δt)

    # Computing momentum flux boundary conditions
    @apply_regionally compute_momentum_flux_bcs!(model)

    # Advance the free surface
    compute_free_surface_tendency!(grid, model, model.free_surface)
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)

    # Update transport velocities
    compute_transport_velocities!(model, model.free_surface)

    # Computing tracer tendencies
    @apply_regionally begin
        compute_tracer_tendencies!(model)

        # Advance grid and velocities
        ab2_step_grid!(model.grid, model, model.vertical_coordinate, Δt, χ)
        ab2_step_velocities!(model.velocities, model, Δt, χ)

        # Correct the barotropic mode
        correct_barotropic_mode!(model, Δt)

        # TODO: fill halo regions for horizontal velocities should be here before the tracer update.
        # Finally advance tracers:
        ab2_step_tracers!(model.tracers, model, Δt, χ)
    end

    return nothing
end

"""
    hydrostatic_ab2_step!(model::HydrostaticFreeSurfaceModel, ::ImplicitFreeSurface, grid, Δt, callbacks)

The Adams-Bashforth 2nd-order time step for `HydrostaticFreeSurfaceModel` with `ImplicitFreeSurface`.

For implicit free surfaces, a predictor-corrector approach is used:
1. Compute momentum and tracer tendencies
2. Advance grid scaling (for z-star coordinates)
3. Advance velocities using AB2 (predictor step)
4. Solve implicit free surface equation
5. Correct velocities for the updated barotropic pressure gradient
6. Advance tracers using AB2
"""
function hydrostatic_ab2_step!(model, free_surface::ImplicitFreeSurface, grid, Δt, callbacks)
    FT = eltype(grid)
    χ  = convert(FT, model.timestepper.χ)
    Δt = convert(FT, Δt)

    @apply_regionally begin
        parent(model.transport_velocities.u) .= parent(model.velocities.u)
        parent(model.transport_velocities.v) .= parent(model.velocities.v)

        # Computing tendencies...
        compute_momentum_flux_bcs!(model)

        # Finally Substep! Advance grid, tracers, (predictor) momentum
        ab2_step_velocities!(model.velocities, model, Δt, χ)
    end

    # Advancing free surface in preparation for the correction step
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)

    # Correct for the updated barotropic mode
    @apply_regionally begin
        correct_barotropic_mode!(model, Δt)

        # Compute transport velocities
        compute_transport_velocities!(model, free_surface)        
        compute_tracer_tendencies!(model)

        ab2_substep_grid!(model.grid, model, model.vertical_coordinate, Δt, χ)
        ab2_substep_tracers!(model.tracers, model, Δt, χ)
    end

    return nothing
end

#####
##### Step grid
#####

"""
    ab2_step_grid!(grid, model, ::ZCoordinate, Δt, χ)

Update grid scaling factors during an AB2 time step.

Fallback method that does nothing. Extended for `ZStarCoordinate` to update
the vertical grid spacing based on the new free surface height.
"""
ab2_step_grid!(grid, model, ::ZCoordinate, Δt, χ) = nothing

#####
##### Step velocities
#####

"""
    ab2_step_velocities!(velocities, model, Δt, χ)

Advance horizontal velocities `u` and `v` using the AB2 scheme.

Velocities are updated as: `u += Δt * ((3/2 + χ) * Gⁿ - (1/2 + χ) * G⁻)`.

If an implicit solver is configured, implicit vertical diffusion is applied after the explicit step.
"""
function ab2_step_velocities!(velocities, model, Δt, χ)

    for (i, name) in enumerate((:u, :v))
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        velocity_field = model.velocities[name]

        launch!(model.architecture, model.grid, :xyz,
                _ab2_step_field!, velocity_field, Δt, χ, Gⁿ, G⁻)

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

const EmptyNamedTuple = NamedTuple{(),Tuple{}}

hasclosure(closure, ClosureType) = closure isa ClosureType
hasclosure(closure_tuple::Tuple, ClosureType) = any(hasclosure(c, ClosureType) for c in closure_tuple)

ab2_step_tracers!(::EmptyNamedTuple, model, Δt, χ) = nothing

"""
    ab2_step_tracers!(tracers, model, Δt, χ)

Advance tracer fields using the AB2 scheme.

For mutable vertical coordinates (z-star), the evolved quantity is `σ * c` where `σ` is
the grid stretching factor. The update accounts for the change in `σ` between time steps.

If CATKE or TD closures are active, their prognostic tracers (`e`, `ϵ`) are skipped
as they are handled separately. Implicit vertical diffusion is applied if configured.
"""
function ab2_step_tracers!(tracers, model, Δt, χ)

    closure = model.closure
    catke_in_closures = hasclosure(closure, FlavorOfCATKE)
    td_in_closures    = hasclosure(closure, FlavorOfTD)

    # Tracer update kernels
    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))

        if catke_in_closures && tracer_name == :e
            @debug "Skipping AB2 step for e"
        elseif td_in_closures && tracer_name == :ϵ
            @debug "Skipping AB2 step for ϵ"
        elseif td_in_closures && tracer_name == :e
            @debug "Skipping AB2 step for e"
        else
            Gⁿ = model.timestepper.Gⁿ[tracer_name]
            G⁻ = model.timestepper.G⁻[tracer_name]
            tracer_field = tracers[tracer_name]
            grid = model.grid

            FT = eltype(grid)
            launch!(architecture(grid), grid, :xyz, _ab2_step_tracer_field!, tracer_field, grid, convert(FT, Δt), χ, Gⁿ, G⁻)

            implicit_step!(tracer_field,
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

# σθ is the evolved quantity. Once σⁿ⁺¹ is known we can retrieve θⁿ⁺¹
@kernel function _ab2_step_tracer_field!(θ, grid, Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    FT = eltype(χ)
    α = 3*one(FT)/2 + χ
    β = 1*one(FT)/2 + χ

    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    σᶜᶜ⁻ = σ⁻(i, j, k, grid, Center(), Center(), Center())

    @inbounds begin
        ∂t_σθ = α * Gⁿ[i, j, k] - β * G⁻[i, j, k]
        θ[i, j, k] = (σᶜᶜ⁻ * θ[i, j, k] + Δt * ∂t_σθ) / σᶜᶜⁿ
    end
end
