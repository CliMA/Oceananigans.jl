using Oceananigans.Utils: work_layout
using Oceananigans.Architectures: device
using KernelAbstractions.Extras.LoopInfo: @unroll

import Oceananigans.TimeSteppers: cache_previous_tendencies!

""" Store source terms for `uh`, `vh`, and `h`. """
@kernel function _cache_solution_tendencies!(G⁻, grid, G⁰)
    i, j, k = @index(Global, NTuple)
    @unroll for t in 1:3
        @inbounds G⁻[t][i, j, k] = G⁰[t][i, j, k]
    end
end

""" Store source terms for `u`, `v`, and `w`. """
@kernel function _cache_field_tendencies!(G⁻, grid, G⁰)
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = G⁰[i, j, k]
end

""" Store previous source terms before updating them. """
function cache_previous_tendencies!(model::ShallowWaterModel)
    _cache_solution!, _ = configure_kernel(model.architecture, model.grid, :xyz, _cache_solution_tendencies!)
    _cache_tracer!, _ = configure_kernel(model.architecture, model.grid, :xyz, _cache_field_tendencies!)

    _cache_solution!(model.timestepper.G⁻, model.grid, model.timestepper.Gⁿ)

    # Tracer fields
    for i in 4:length(model.timestepper.G⁻)
        @inbounds Gc⁻ = model.timestepper.G⁻[i]
        @inbounds Gc⁰ = model.timestepper.Gⁿ[i]
        _cache_tracer!(Gc⁻, model.grid, Gc⁰)
    end

    return nothing
end

