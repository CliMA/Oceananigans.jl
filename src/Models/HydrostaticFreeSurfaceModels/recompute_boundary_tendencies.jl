function recompute_boundary_tendencies(model)
    grid = model.grid
    arch = architecture(grid)

    # What shall we do with w, p and κ???

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    size_x = (Hx, Ny, Nz)
    size_y = (Nx, Hy, Nz-2Hz)
    size_z = (Nx-2Hx, Ny-2Hy, Hz)

    offsetᴸx = (0,  0,  0)
    offsetᴸy = (0,  0,  Hz)
    offsetᴸz = (Hx, Hy, 0)
    offsetᴿx = (Nx-Hx, 0,      0)
    offsetᴿy = (0,     Ny-Hy, Hz)
    offsetᴿz = (Hx,    Hy,    Nz-Hz)

    sizes   = (size_x, size_y, size_z, size_x, size_y, size_z)
    offsets = (offsetᴸx, offsetᴸy, offsetᴸz, offsetᴿx, offsetᴿy, offsetᴿz)

    u_immersed_bc = immersed_boundary_condition(model.velocities.u)
    v_immersed_bc = immersed_boundary_condition(model.velocities.v)

    start_momentum_kernel_args = (grid,
                                  model.advection.momentum,
                                  model.coriolis,
                                  model.closure)

    end_momentum_kernel_args = (model.velocities,
                                model.free_surface,
                                model.tracers,
                                model.buoyancy,
                                model.diffusivity_fields,
                                model.pressure.pHY′,
                                model.auxiliary_fields,
                                model.forcing,
                                model.clock)

    u_kernel_args = tuple(start_momentum_kernel_args..., u_immersed_bc, end_momentum_kernel_args...)
    v_kernel_args = tuple(start_momentum_kernel_args..., v_immersed_bc, end_momentum_kernel_args...)
    
    only_active_cells = use_only_active_cells(grid)

    for (kernel_size, kernel_offsets) in zip(sizes, offsets)
        launch!(arch, grid, kernel_size,
                calculate_hydrostatic_free_surface_Gu!, model.timestepper.Gⁿ.u, kernel_offsets, u_kernel_args...;
                only_active_cells)
    
        launch!(arch, grid, kernel_size,
                calculate_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, kernel_offsets, v_kernel_args...;
                only_active_cells)
    end

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

        args = (c_kernel_function,
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

        for (kernel_size, kernel_offsets) in zip(sizes, offsets)
            launch!(arch, grid, kernel_size, calculate_hydrostatic_free_surface_Gc!, c_tendency, kernel_offsets, args...)
        end
    end
end