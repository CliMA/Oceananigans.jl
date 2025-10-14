using KernelAbstractions: @index, @kernel
using Oceananigans: prognostic_fields
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: launch!

import Oceananigans.TimeSteppers: cache_previous_tendencies!

#####
##### Storing previous tendencies for the AB2 update
#####

""" Store source terms for `η`. """
@kernel function _cache_free_surface_tendency!(Gη⁻, grid, Gη⁰)
    i, j = @index(Global, NTuple)
    @inbounds Gη⁻[i, j, grid.Nz+1] = Gη⁰[i, j, grid.Nz+1]
end

cache_free_surface_tendency!(free_surface, model) = nothing

function cache_free_surface_tendency!(::ExplicitFreeSurface, model)
    launch!(model.architecture, model.grid, :xy,
            _cache_free_surface_tendency!,
            model.timestepper.G⁻.η,
            model.grid,
            model.timestepper.Gⁿ.η)
end

@kernel function _cache_field_tendencies!(G⁻, G⁰)
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = G⁰[i, j, k]
end

""" Store previous source terms before updating them. """
function cache_previous_tendencies!(model::HydrostaticFreeSurfaceModel)
    prognostic_field_names = keys(prognostic_fields(model))
    three_dimensional_prognostic_field_names = filter(name -> name != :η, prognostic_field_names)

    closure = model.closure
    catke_in_closures = hasclosure(closure, FlavorOfCATKE)
    td_in_closures    = hasclosure(closure, FlavorOfTD)

    for field_name in three_dimensional_prognostic_field_names

        if catke_in_closures && field_name == :e
            @debug "Skipping store tendencies for e"
        elseif td_in_closures && field_name == :ϵ
            @debug "Skipping store tendencies for ϵ"
        elseif td_in_closures && field_name == :e
            @debug "Skipping store tendencies for e"
        else
            launch!(model.architecture, model.grid, :xyz,
                    _cache_field_tendencies!,
                    model.timestepper.G⁻[field_name],
                    model.timestepper.Gⁿ[field_name])
        end
    end

    cache_free_surface_tendency!(model.free_surface, model)

    return nothing
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
