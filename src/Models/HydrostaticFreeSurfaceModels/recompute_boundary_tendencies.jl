import Oceananigans.Distributed: recompute_boundary_tendencies!

# We assume here that top/bottom BC are always synched (no partitioning in z)
function recompute_boundary_tendencies!(model::HydrostaticFreeSurfaceModel)
    grid = model.grid
    arch = architecture(grid)

    # We need new values for `w`, `p` and `κ`
    recompute_auxiliaries!(model, grid, arch)

    sizes, offsets = size_tendency_kernel(grid, arch)

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
    
    sizes, offs = size_w_kernel(grid, arch)

    for (kernel_size, kernel_offsets) in zip(sizes, offs)
        compute_w_from_continuity!(model.velocities, arch, grid; kernel_size, kernel_offsets)
    end

    sizes, offs = size_p_kernel(grid, arch)

    for (kernel_size, kernel_offsets) in zip(sizes, offs)
        update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers; kernel_size, kernel_offsets)
    end

    sizes, offs = size_κ_kernel(grid, arch)

    for (kernel_size, kernel_offsets) in zip(sizes, offs)
        calculate_diffusivities!(model.diffusivity_fields, model.closure, model; kernel_size, kernel_offsets)
    end
end

function size_w_kernel(grid, arch)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    Rx, Ry, _ = arch.ranks

    size_x = (Hx, Ny)
    size_y = (Nx, Hy)

    offsᴸx = (-Hx+1, 0)
    offsᴸy = (0, -Hy+1)
    offsᴿx = (Nx-1, 0)
    offsᴿy = (0, Ny-1)

    sizes = (size_x, size_y, size_x, size_y)
    offs  = (offsᴸx, offsᴸy, offsᴿx, offsᴿy)
        
    return return_correct_directions(Rx, Ry, sizes, offs, grid)
end

function size_p_kernel(grid, arch)
    Nx, Ny, _ = size(grid)
    Rx, Ry, _ = arch.ranks

    size_x = (1, Ny)
    size_y = (Nx, 1)

    offsᴸx = (-1, 0)
    offsᴸy = (0, -1)
    offsᴿx = (Nx, 0)
    offsᴿy = (0, Ny)

    sizes = (size_x, size_y, size_x, size_y)
    offs  = (offsᴸx, offsᴸy, offsᴿx, offsᴿy)
        
    return return_correct_directions(Rx, Ry, sizes, offs, grid)
end

function size_κ_kernel(grid, arch)
    Nx, Ny, Nz = size(grid)
    Rx, Ry, _  = arch.ranks

    size_x = (1, Ny, Nz)
    size_y = (Nx, 1, Nz)

    offsᴸx = (-1,  0, 0)
    offsᴸy = (0,  -1, 0)
    offsᴿx = (Nx,  0, 0)
    offsᴿy = (0,  Ny, 0)

    sizes = (size_x, size_y, size_x, size_y)
    offs  = (offsᴸx, offsᴸy, offsᴿx, offsᴿy)
        
    return return_correct_directions(Rx, Ry, sizes, offs, grid)
end

function size_tendency_kernel(grid, arch)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, _  = halo_size(grid)
    Rx, Ry, _  = arch.ranks
    
    size_x = (Hx, Ny, Nz)
    size_y = (Nx, Hy, Nz)
    
    offsᴸx = (0,  0,  0)
    offsᴸy = (0,  0,  0)
    offsᴿx = (Nx-Hx, 0,     0)
    offsᴿy = (0,     Ny-Hy, 0)

    sizes = (size_x, size_y, size_x, size_y)
    offs  = (offsᴸx, offsᴸy, offsᴿx, offsᴿy)
        
    return return_correct_directions(Rx, Ry, sizes, offs, grid)
end

using Oceananigans.Operators: XFlatGrid, YFlatGrid

function return_correct_directions(Rx, Ry, s, o, grid) 
    include_x = !isa(grid, XFlatGrid) && (Rx != 1)
    include_y = !isa(grid, YFlatGrid) && (Ry != 1)

    if include_x && include_y
        return s, o
    elseif include_x && !(include_y)
        return (s[1], s[3]), (o[1], o[3])
    elseif !(include_x) && include_y
        return (s[2], s[4]), (o[2], o[4])
    else
        return (), ()
    end
end

