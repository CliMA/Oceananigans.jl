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
    cache_previous_tendencies!(model, Val(keys(prognostic_fields(model))))
    return nothing
end

@inline cache_previous_tendencies!(model, ::Val{()}) = nothing

@inline function cache_previous_tendencies!(model, ::Val{names}) where names
    name = first(names)
    launch!(model.architecture, model.grid, :xyz, _cache_field_tendencies!,
            model.timestepper.G⁻[name],
            model.timestepper.Gⁿ[name])
    cache_previous_tendencies!(model, Val(Base.tail(names)))
    return nothing
end

# Snapshot `wⁿ`, before any field is stepped, for use in the `implicit_step!`.
function implicit_advecting_velocities(model)
    cache_advecting_vertical_velocity!(model.advecting_vertical_velocity, model.velocities)
    return advecting_velocities(model)
end

cache_advecting_vertical_velocity!(::Nothing, velocities) = nothing
cache_advecting_vertical_velocity!(w, velocities) = parent(w) .= parent(velocities.w)

@inline advecting_velocities(model) = advecting_velocities(model, model.advecting_vertical_velocity)
@inline advecting_velocities(model, ::Nothing) = model.velocities
@inline advecting_velocities(model, w) = (; w)
