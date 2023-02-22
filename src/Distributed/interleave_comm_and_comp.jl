function complete_communication_and_compute_boundary(model, grid::DistributedGrid)

    arch = architecture(grid)

    MPI.Waitall(arch.mpi_requests)
    empty!(arch.mpi_requests)

    for side in (:west_and_east, :south_and_north, :bottom_and_top)
        for field in prognostic_fields(model)
            recv_from_buffers!(field.data, field.boundary_buffers, grid, Val(side))    
        end
    end

    # HERE we have to put fill_eventual_halo_corners
    recompute_boundary_tendencies(model)

    return nothing
end

function recompute_boundary_tendencies(model)

    arch = model.architecture
    grid = model.grid

    recompute_calculate_hydrostatic_momentum_tendencies!(model, model.velocities)

    top_tracer_bcs = top_tracer_boundary_conditions(grid, model.tracers)

    only_active_cells = use_only_active_cells(grid)

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))
        @inbounds c_tendency = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection = model.advection[tracer_name]
        @inbounds c_forcing = model.forcing[tracer_name]
        @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])

        c_kernel_function, closure, diffusivity_fields = tracer_tendency_kernel_function(model,
                                                                                         Val(tracer_name),
                                                                                         model.closure,
                                                                                         model.diffusivity_fields)

        args = (calculate_hydrostatic_free_surface_Gc!,
                c_tendency,
                c_kernel_function,
                grid,
                Val(tracer_index),
                c_advection,
                closure,
                c_immersed_bc,
                model.buoyancy,
                model.velocities,
                model.free_surface,
                model.tracers,
                top_tracer_bcs,
                diffusivity_fields,
                model.auxiliary_fields,
                c_forcing,
                model.clock)

        launch!(arch, grid, :xyz, args...; only_active_cells)
    end
end

function compute_full_w_and_pressures!(model)

