""" Store source terms for `uh`, `vh`, `h`, and tracers. """
@kernel function _cache_field_tendencies!(G⁻, grid, G⁰)
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = G⁰[i, j, k]
end

"""
$(TYPEDSIGNATURES)

Store the current tendencies `Gⁿ` into `G⁻` for solution fields (`uh`, `vh`, `h`) and tracers.

This function is called after advancing the model state but before computing new tendencies,
preserving the tendencies needed for a :RungeKutta3 timestepper.
"""
function Oceananigans.TimeSteppers.cache_previous_tendencies!(model::ShallowWaterModel)
    _cache_field!, _ = configure_kernel(model.architecture, model.grid, :xyz, _cache_field_tendencies!)

    # One launch per field: fields have different static sizes,
    # so a NamedTuple of them can't be indexed in a kernel
    for (G⁻, Gⁿ) in zip(model.timestepper.G⁻, model.timestepper.Gⁿ)
        _cache_field!(G⁻, model.grid, Gⁿ)
    end

    return nothing
end
