using KernelAbstractions: @index, @kernel, NoneEvent

using Oceananigans.TimeSteppers:  store_field_tendencies!

using Oceananigans: prognostic_fields
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Architectures: device_event

using Oceananigans.Utils: launch!

import Oceananigans.TimeSteppers: store_tendencies!

""" Store source terms for `η`. """
@kernel function _store_free_surface_tendency!(Gη⁻, grid, Gη⁰)
    i, j = @index(Global, NTuple)
    @inbounds Gη⁻[i, j, grid.Nz+1] = Gη⁰[i, j, grid.Nz+1]
end

store_free_surface_tendency!(free_surface, model, barrier) = NoneEvent()

function store_free_surface_tendency!(::ExplicitFreeSurface, model, barrier)

    event = launch!(model.architecture, model.grid, :xy,
                    _store_free_surface_tendency!,
                    model.timestepper.G⁻.η,
                    model.grid,
                    model.timestepper.Gⁿ.η,
                    dependencies = barrier)

    return event
end

""" Store previous source terms before updating them. """
function store_tendencies!(model::HydrostaticFreeSurfaceModel)
    prognostic_field_names = keys(prognostic_fields(model))
    three_dimensional_prognostic_field_names = filter(name -> name != :η, prognostic_field_names)

    for field_name in three_dimensional_prognostic_field_names

        launch!(model.architecture, model.grid, :xyz,
                store_field_tendencies!,
                model.timestepper.G⁻[field_name],
                model.grid,
                model.timestepper.Gⁿ[field_name])

    end

    store_free_surface_tendency!(model.free_surface, model)

    return nothing
end
