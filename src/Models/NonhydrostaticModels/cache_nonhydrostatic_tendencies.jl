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
$(TYPEDSIGNATURES)

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

# `w` advects itself so we need to pass a clean `w` without it overwriting itself. Reuse `G⁻.w` that is free between substep kernels.
@inline function implicit_advecting_velocities(model, name)
    (name === :w && needs_implicit_solver(model.advection)) || return model.velocities
    w = model.timestepper.G⁻.w
    parent(w) .= parent(model.velocities.w)
    return (; w)
end
