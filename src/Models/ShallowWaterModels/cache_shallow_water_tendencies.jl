""" Store source terms for `uh`, `vh`, and `h`. """
@kernel function _cache_solution_tendencies!(G⁻, grid, G⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        G⁻[1][i, j, k] = G⁰[1][i, j, k]
        G⁻[2][i, j, k] = G⁰[2][i, j, k]
        G⁻[3][i, j, k] = G⁰[3][i, j, k]
    end
end

""" Store source terms for `u`, `v`, and `w`. """
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
function TimeSteppers.cache_previous_tendencies!(model::ShallowWaterModel)
    _cache_field!, _ = configure_kernel(model.architecture, model.grid, :xyz, _cache_field_tendencies!)

    # One launch per field: fields have different static sizes,
    # so a NamedTuple of them can't be indexed in a kernel
    for (G⁻, Gⁿ) in zip(model.timestepper.G⁻, model.timestepper.Gⁿ)
        _cache_field!(G⁻, model.grid, Gⁿ)
    end

    return nothing
end
