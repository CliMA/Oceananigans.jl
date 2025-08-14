using Oceananigans.Fields: location, instantiated_location
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: get_active_cells_map, get_active_column_map

import Oceananigans.TimeSteppers: split_rk3_substep!, _euler_substep_field!, _split_rk3_average_field!, cache_previous_fields!

function split_rk3_substep!(model::HydrostaticFreeSurfaceModel, Δt, γⁿ, ζⁿ)

    grid         = model.grid
    timestepper  = model.timestepper
    free_surface = model.free_surface

    compute_free_surface_tendency!(grid, model, free_surface)

    @apply_regionally begin
        scale_by_stretching_factor!(model.timestepper.Gⁿ, model.tracers, model.grid)
        rk3_substep_grid!(grid, model, model.vertical_coordinate, Δt, γⁿ, ζⁿ)
        rk3_substep_velocities!(model.velocities, model, Δt, γⁿ, ζⁿ)
        rk3_substep_tracers!(model.tracers, model, Δt, γⁿ, ζⁿ)
    end

    # Full step for Implicit and Split-Explicit, substep for Explicit
    step_free_surface!(free_surface, model, timestepper, Δt)

    # Average free surface variables in the second stage
    if model.clock.stage == 2 
        @apply_regionally rk3_average_free_surface!(free_surface, grid, timestepper, γⁿ, ζⁿ)
    end
    
    return nothing
end

rk3_average_free_surface!(free_surface, args...) = nothing

function rk3_average_free_surface!(free_surface::ImplicitFreeSurface, grid, timestepper, γⁿ, ζⁿ)
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)

    ηⁿ⁻¹ = timestepper.Ψ⁻.η
    ηⁿ   = free_surface.η
    params = KernelParameters(1:Nx, 1:Ny, Nz+1:Nz+1)

    launch!(arch, grid, params, _split_rk3_average_field!, ηⁿ, γⁿ, ζⁿ, ηⁿ⁻¹)

    return nothing
end

function rk3_average_free_surface!(free_surface::SplitExplicitFreeSurface, grid, timestepper, γⁿ, ζⁿ)

    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)

    Uⁿ⁻¹ = timestepper.Ψ⁻.U
    Vⁿ⁻¹ = timestepper.Ψ⁻.V
    Uⁿ   = free_surface.barotropic_velocities.U
    Vⁿ   = free_surface.barotropic_velocities.V
    params = KernelParameters(1:Nx, 1:Ny, Nz+1:Nz+1)
    
    launch!(arch, grid, params, _split_rk3_average_field!, Uⁿ, γⁿ, ζⁿ, Uⁿ⁻¹)
    launch!(arch, grid, params, _split_rk3_average_field!, Vⁿ, γⁿ, ζⁿ, Vⁿ⁻¹)

    return nothing
end

#####
##### Time stepping in each substep
#####

function rk3_substep_velocities!(velocities, model, Δt, γⁿ, ζⁿ)

    grid = model.grid
    FT = eltype(grid)

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
                       Δt)

        if model.clock.stage > 1 
            launch!(architecture(grid), grid, :xyz,
                    _split_rk3_average_field!, velocity_field, γⁿ, ζⁿ, Ψ⁻)
        end
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
                       Δt)

        if model.clock.stage > 1 
            launch!(architecture(grid), grid, :xyz,
                    _split_rk3_average_tracer_field!, c, grid, γⁿ, ζⁿ, Ψ⁻)
        end
    end

    return nothing
end

#####
##### Tracer update in mutable vertical coordinates
#####


# σc is the evolved quantity, so tracer fields need to be evolved
# accounting for the stretching factors from the new and the previous time step.
@kernel function _euler_substep_tracer_field!(c, grid, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    σᶜᶜ⁻ = σ⁻(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (σᶜᶜ⁻ * c[i, j, k] + Δt * Gⁿ[i, j, k]) / σᶜᶜⁿ
end

@kernel function _split_rk3_average_tracer_field!(c, grid, γⁿ, ζⁿ, c⁻)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = ζⁿ * c⁻[i, j, k] / σᶜᶜⁿ + γⁿ * c[i, j, k]
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

    if grid isa MutableGridOfSomeKind && model.vertical_coordinate isa ZStarCoordinate
        # We need to cache the surface height somewhere!
        parent(model.vertical_coordinate.storage) .= parent(model.grid.z.ηⁿ)
    end

    return nothing
end
