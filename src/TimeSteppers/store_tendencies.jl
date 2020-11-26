using Oceananigans.Grids: AbstractGrid

""" Store source terms for `u`, `v`, and `w`. """
@kernel function store_field_tendencies!(G⁻, grid::AbstractGrid{FT}, G⁰) where FT
    i, j, k = @index(Global, NTuple)
    G⁻[i, j, k] = G⁰[i, j, k]
end

""" Store previous source terms before updating them. """
function store_tendencies!(model)

    barrier = Event(device(model.architecture))

    workgroup, worksize = work_layout(model.grid, :xyz)

    model_fields = fields(model)

    store_field_tendencies_kernel! = store_field_tendencies!(device(model.architecture), workgroup, worksize)

    events = []

    for (i, field) in enumerate(model_fields)

        field_event = store_field_tendencies_kernel!(model.timestepper.G⁻,
                                                     model.grid,
                                                     model.timestepper.Gⁿ,
                                                     dependencies=barrier)

        push!(events, field_event)
    end

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    return nothing
end
