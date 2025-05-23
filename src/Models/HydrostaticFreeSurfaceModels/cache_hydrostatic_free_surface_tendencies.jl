using KernelAbstractions: @index, @kernel

using Oceananigans.TimeSteppers: _cache_field_tendencies!

using Oceananigans: prognostic_fields
using Oceananigans.Grids: AbstractGrid

using Oceananigans.Utils: launch!

import Oceananigans.TimeSteppers: cache_previous_tendencies!

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

