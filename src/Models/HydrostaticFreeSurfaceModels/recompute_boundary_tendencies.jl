import Oceananigans.Distributed: recompute_boundary_tendencies!

# We assume here that top/bottom BC are always synched (no partitioning in z)
function recompute_boundary_tendencies!(model::HydrostaticFreeSurfaceModel)
    grid = model.grid
    arch = architecture(grid)

    # We need new values for `w`, `p` and `κ`
    recompute_auxiliaries!(model, grid, arch)

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    size_x = (Hx, Ny, Nz)
    size_y = (Nx, Hy, Nz)

    offsetᴸx = (0,  0,  0)
    offsetᴸy = (0,  0,  0)
    offsetᴿx = (Nx-Hx, 0,     0)
    offsetᴿy = (0,     Ny-Hy, 0)

    sizes   = (size_x,     size_y,   size_x,   size_y)
    offsets = (offsetᴸx, offsetᴸy, offsetᴿx, offsetᴿy)

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
    
    for (kernel_size, kernel_offsets) in zip(sizes, offsets)
        launch!(arch, grid, kernel_size,
                calculate_hydrostatic_free_surface_Gu!, model.timestepper.Gⁿ.u, kernel_offsets, u_kernel_args...)
    
        launch!(arch, grid, kernel_size,
                calculate_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, kernel_offsets, v_kernel_args...)
        
        launch!(arch, grid, kernel_size[1:2],
                calculate_hydrostatic_free_surface_Gη!, model.timestepper.Gⁿ.η, kernel_offsets[1:2],
                grid, model.velocities, model.free_surface, model.tracers, model.auxiliary_fields, model.forcing,
                model.clock)
    end

    top_tracer_bcs = top_tracer_boundary_conditions(grid, model.tracers)

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

function recompute_auxiliaries!(model, grid, arch)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    size_x = (Hx, Ny)
    size_y = (Nx, Hy)

    offsetᴸx = (-Hx+1,  0)
    offsetᴸy = (0,  -Hy+1)
    offsetᴿx = (Nx-1, 0)
    offsetᴿy = (0, Ny-1)

    sizes   = (size_x,     size_y,   size_x,   size_y)
    offsets = (offsetᴸx, offsetᴸy, offsetᴿx, offsetᴿy)

    for (kernel_size, kernel_offsets) in zip(sizes, offsets)
        compute_w_from_continuity!(model.velocities, arch, grid; kernel_size, kernel_offsets)
    end

    size_x = (1, Ny)
    size_y = (Nx, 1)

    offsetᴸx = (-1,  0)
    offsetᴸy = (0,  -1)
    offsetᴿx = (Nx,  0)
    offsetᴿy = (0,  Ny)

    sizes   = (size_x,     size_y,   size_x,   size_y)
    offsets = (offsetᴸx, offsetᴸy, offsetᴿx, offsetᴿy)


    for (kernel_size, kernel_offsets) in zip(sizes, offsets)
        update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers; kernel_size, kernel_offsets)
    end

    size_x = (1, Ny, Nz)
    size_y = (Nx, 1, Nz)

    offsetᴸx = (-1,  0, 0)
    offsetᴸy = (0,  -1, 0)
    offsetᴿx = (Nx,  0, 0)
    offsetᴿy = (0,  Ny, 0)

    sizes   = (size_x,     size_y,   size_x,   size_y)
    offsets = (offsetᴸx, offsetᴸy, offsetᴿx, offsetᴿy)

    for (kernel_size, kernel_offsets) in zip(sizes, offsets)
        calculate_diffusivities!(model.diffusivity_fields, model.closure, model; kernel_size, kernel_offsets)
    end
end