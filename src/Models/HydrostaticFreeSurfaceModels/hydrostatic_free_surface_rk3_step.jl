using Oceananigans.Fields: location, instantiated_location
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: get_active_cells_map, get_active_column_map

import Oceananigans.TimeSteppers: split_rk3_substep!, _split_rk3_substep_field!, cache_previous_fields!

function split_rk3_substep!(model::HydrostaticFreeSurfaceModel, Δt, γⁿ, ζⁿ)

    grid         = model.grid
    timestepper  = model.timestepper
    free_surface = model.free_surface

    compute_free_surface_tendency!(grid, model, free_surface)

    rk3_substep_velocities!(model.velocities, model, Δt, γⁿ, ζⁿ)
    rk3_substep_tracers!(model.tracers, model, Δt, γⁿ, ζⁿ)

    # Full step for Implicit and Split-Explicit, substep for Explicit
    step_free_surface!(free_surface, model, timestepper, Δt)

    # Average free surface variables in the second stage
    if model.clock.stage == 2
        rk3_average_free_surface!(free_surface, grid, timestepper, γⁿ, ζⁿ)
    end

    return nothing
end

rk3_average_free_surface!(free_surface, args...) = nothing

function rk3_average_free_surface!(free_surface::ImplicitFreeSurface, grid, timestepper, γⁿ, ζⁿ)
    arch = architecture(grid)

    ηⁿ⁻¹ = timestepper.Ψ⁻.η
    ηⁿ   = free_surface.η

    launch!(arch, grid, :xy, _rk3_average_free_surface!, ηⁿ, grid, ηⁿ⁻¹, γⁿ, ζⁿ)

    return nothing
end

function rk3_average_free_surface!(free_surface::SplitExplicitFreeSurface, grid, timestepper, γⁿ, ζⁿ)

    arch = architecture(grid)

    Uⁿ⁻¹ = timestepper.Ψ⁻.U
    Vⁿ⁻¹ = timestepper.Ψ⁻.V
    ηⁿ⁻¹ = timestepper.Ψ⁻.η
    Uⁿ   = free_surface.barotropic_velocities.U
    Vⁿ   = free_surface.barotropic_velocities.V
    ηⁿ   = free_surface.η

    launch!(arch, grid, :xy, _rk3_average_free_surface!, Uⁿ, grid, Uⁿ⁻¹, γⁿ, ζⁿ)
    launch!(arch, grid, :xy, _rk3_average_free_surface!, Vⁿ, grid, Vⁿ⁻¹, γⁿ, ζⁿ)

    return nothing
end

@kernel function _rk3_average_free_surface!(η, grid, η⁻, γⁿ, ζⁿ)
    i, j = @index(Global, NTuple)
    k = grid.Nz + 1
    @inbounds η[i, j, k] = ζⁿ * η⁻[i, j, k] + γⁿ * η[i, j, k]
end

#####
##### Time stepping in each substep
#####

function rk3_substep_velocities!(velocities, model, Δt, γⁿ, ζⁿ)

    for name in (:u, :v)
        Gⁿ = model.timestepper.Gⁿ[name]
        Ψ⁻ = model.timestepper.Ψ⁻[name]
        velocity_field = velocities[name]

        launch!(model.architecture, model.grid, :xyz,
                _split_rk3_substep_field!, velocity_field, Δt, γⁿ, ζⁿ, Gⁿ, Ψ⁻)

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

rk3_substep_tracers!(::EmptyNamedTuple, model, Δt, γⁿ, ζⁿ) = nothing

function rk3_substep_tracers!(tracers, model, Δt, γⁿ, ζⁿ)

    closure = model.closure
    grid = model.grid
    FT = eltype(grid)

    # Tracer update kernels
    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))

        Gⁿ = model.timestepper.Gⁿ[tracer_name]
        Ψ⁻ = model.timestepper.Ψ⁻[tracer_name]
        θ  = tracers[tracer_name]
        closure = model.closure

        launch!(architecture(grid), grid, :xyz,
                _split_rk3_substep_tracer_field!, θ, grid, convert(FT, Δt), γⁿ, ζⁿ, Gⁿ, Ψ⁻)

        implicit_step!(θ,
                       model.timestepper.implicit_solver,
                       closure,
                       model.diffusivity_fields,
                       Val(tracer_index),
                       model.clock,
                       Δt)
    end

    return nothing
end

#####
##### Tracer update in mutable vertical coordinates
#####

# σθ is the evolved quantity.
# We store temporarily σθ in θ. Once σⁿ⁺¹ is known we can retrieve θⁿ⁺¹
# with the `unscale_tracers!` function. Ψ⁻ is the previous tracer already scaled
# by the vertical coordinate scaling factor: ψ⁻ = σ * θ
@kernel function _split_rk3_substep_tracer_field!(θ, grid, Δt, γⁿ, ζⁿ, Gⁿ, Ψ⁻)
    i, j, k = @index(Global, NTuple)

    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds θ[i, j, k] = ζⁿ * Ψ⁻[i, j, k] + γⁿ * σᶜᶜⁿ * (θ[i, j, k] + Δt * Gⁿ[i, j, k])
end

# We store temporarily σθ in θ.
# The unscaled θ will be retrieved with `unscale_tracers!`
@kernel function _split_rk3_substep_tracer_field!(θ, grid, Δt, ::Nothing, ::Nothing, Gⁿ, Ψ⁻)
    i, j, k = @index(Global, NTuple)
    @inbounds θ[i, j, k] = Ψ⁻[i, j, k] + Δt * Gⁿ[i, j, k] * σⁿ(i, j, k, grid, Center(), Center(), Center())
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

    return nothing
end
