using Oceananigans: prognostic_fields
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: launch!

import Oceananigans.TimeSteppers: cache_previous_tendencies!

""" Store source terms for `u`, `v`, and `w`. """
@kernel function _cache_field_tendencies!(G⁻, G⁰)
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = G⁰[i, j, k]
end

"""
    cache_previous_tendencies!(model::NonhydrostaticModel)

Store the current tendencies `Gⁿ` into `G⁻` for all prognostic fields (velocities and tracers).

This function is called after advancing the model state but before computing new tendencies,
preserving the tendencies needed for multi-step time-stepping schemes (:QuasiAdamsBashorth2 and :RungeKutta3)
"""
function cache_previous_tendencies!(model::NonhydrostaticModel)
    model_fields = prognostic_fields(model)

    for field_name in keys(model_fields)
        launch!(model.architecture, model.grid, :xyz, _cache_field_tendencies!,
                model.timestepper.G⁻[field_name],
                model.timestepper.Gⁿ[field_name])
    end

    return nothing
end
