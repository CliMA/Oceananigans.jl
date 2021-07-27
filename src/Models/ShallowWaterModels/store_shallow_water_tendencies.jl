using Oceananigans.Utils: work_layout
using Oceananigans.Architectures: device
using Oceananigans.TimeSteppers: store_tracer_tendency!

import Oceananigans.TimeSteppers: store_tendencies!

""" Store source terms for `uh`, `vh`, and `h`. """
@kernel function store_solution_tendencies!(G⁻, grid, G⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds G⁻.uh[i, j, k] = G⁰.uh[i, j, k]
    @inbounds G⁻.vh[i, j, k] = G⁰.vh[i, j, k]
    @inbounds G⁻.h[i, j, k]  = G⁰.h[i, j, k]
end

""" Store previous source terms before updating them. """
function store_tendencies!(model::ShallowWaterModel)

    barrier = Event(device(model.architecture))

    workgroup, worksize = work_layout(model.architecture, model.grid, :xyz)

    store_solution_tendencies_kernel! = store_solution_tendencies!(device(model.architecture), workgroup, worksize)
    store_tracer_tendency_kernel! = store_tracer_tendency!(device(model.architecture), workgroup, worksize)

    solution_event = store_solution_tendencies_kernel!(model.timestepper.G⁻,
                                                       model.grid,
                                                       model.timestepper.Gⁿ,
                                                       dependencies=barrier)

    events = [solution_event]

    # Tracer fields
    for i in 4:length(model.timestepper.G⁻)
        @inbounds Gc⁻ = model.timestepper.G⁻[i]
        @inbounds Gc⁰ = model.timestepper.Gⁿ[i]
        tracer_event = store_tracer_tendency_kernel!(Gc⁻, model.grid, Gc⁰, dependencies=barrier)
        push!(events, tracer_event)
    end

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    return nothing
end

