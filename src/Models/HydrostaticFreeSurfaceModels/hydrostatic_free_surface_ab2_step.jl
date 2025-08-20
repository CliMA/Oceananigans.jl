using Oceananigans.Fields: location
using Oceananigans.TimeSteppers: ab2_step_field!
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: get_active_cells_map, get_active_column_map

import Oceananigans.TimeSteppers: ab2_step!

#####
##### Step everything
#####

function ab2_step!(model::HydrostaticFreeSurfaceModel, Δt)

    grid = model.grid
    compute_free_surface_tendency!(grid, model, model.free_surface)

    FT = eltype(grid)
    χ  = convert(FT, model.timestepper.χ)
    Δt = convert(FT, Δt)

    # Step locally velocity and tracers
    @apply_regionally begin
        scale_by_stretching_factor!(model.timestepper.Gⁿ, model.tracers, model.grid)
        ab2_step_grid!(model.grid, model, model.vertical_coordinate, Δt, χ)
        ab2_step_velocities!(model.velocities, model, Δt, χ)
        ab2_step_tracers!(model.tracers, model, Δt, χ)
    end

    step_free_surface!(model.free_surface, model, model.timestepper, Δt)

    return nothing
end

#####
##### Step velocities
#####

function ab2_step_velocities!(velocities, model, Δt, χ)

    for (i, name) in enumerate((:u, :v))
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        velocity_field = model.velocities[name]

        launch!(model.architecture, model.grid, :xyz,
                ab2_step_field!, velocity_field, Δt, χ, Gⁿ, G⁻)

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

const EmptyNamedTuple = NamedTuple{(),Tuple{}}

hasclosure(closure, ClosureType) = closure isa ClosureType
hasclosure(closure_tuple::Tuple, ClosureType) = any(hasclosure(c, ClosureType) for c in closure_tuple)

ab2_step_tracers!(::EmptyNamedTuple, model, Δt, χ) = nothing

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
            closure = model.closure
            grid = model.grid

            FT = eltype(grid)
            launch!(architecture(grid), grid, :xyz, _ab2_step_tracer_field!, tracer_field, grid, convert(FT, Δt), χ, Gⁿ, G⁻)

            implicit_step!(tracer_field,
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

# σθ is the evolved quantity. Once σⁿ⁺¹ is known we can retrieve θⁿ⁺¹
@kernel function _ab2_step_tracer_field!(θ, grid, Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    FT = eltype(χ)
    α = convert(FT, 1.5) + χ
    β = convert(FT, 0.5) + χ

    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    σᶜᶜ⁻ = σ⁻(i, j, k, grid, Center(), Center(), Center())

    @inbounds begin
        ∂t_σθ = α * Gⁿ[i, j, k] - β * G⁻[i, j, k]
        θ[i, j, k] = (σᶜᶜ⁻ * θ[i, j, k] + Δt * ∂t_σθ) / σᶜᶜⁿ
    end
end
