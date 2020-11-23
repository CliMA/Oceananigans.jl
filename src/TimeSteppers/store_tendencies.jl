using Oceananigans.Grids: AbstractGrid

""" Store source terms for `u`, `v`, and `w`. """
@kernel function store_velocity_tendencies!(G⁻, grid::AbstractGrid{FT}, G⁰) where FT
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻.u[i, j, k] = G⁰.u[i, j, k]
    @inbounds G⁻.v[i, j, k] = G⁰.v[i, j, k]
    @inbounds G⁻.w[i, j, k] = G⁰.w[i, j, k]
end

""" Store previous source terms for a tracer before updating them. """
@kernel function store_tracer_tendency!(Gc⁻, grid::AbstractGrid{FT}, Gc⁰) where FT
    i, j, k = @index(Global, NTuple)
    @inbounds Gc⁻[i, j, k] = Gc⁰[i, j, k]
end

""" Store previous source terms before updating them. """
function store_tendencies!(model)

    barrier = Event(device(model.architecture))

    workgroup, worksize = work_layout(model.grid, :xyz)

    store_velocity_tendencies_kernel! = store_velocity_tendencies!(device(model.architecture), workgroup, worksize)
    store_tracer_tendency_kernel! = store_tracer_tendency!(device(model.architecture), workgroup, worksize)

    velocities_event = store_velocity_tendencies_kernel!(model.timestepper.G⁻,
                                                         model.grid,
                                                         model.timestepper.Gⁿ,
                                                         dependencies=barrier)

    events = [velocities_event]

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
