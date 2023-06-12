import Oceananigans.Distributed: compute_boundary_tendencies!
using Oceananigans.Utils: worktuple, offsets
using Oceananigans.TurbulenceClosures: required_halo_size

# We assume here that top/bottom BC are always synched (no partitioning in z)
function compute_boundary_tendencies!(model::HydrostaticFreeSurfaceModel)
    grid = model.grid
    arch = architecture(grid)

    # We need new values for `w`, `p` and `κ`
    recompute_auxiliaries!(model, grid, arch)

    kernel_parameters = boundary_tendency_kernel_parameters(grid, arch)

    u_immersed_bc = immersed_boundary_condition(model.velocities.u)
    v_immersed_bc = immersed_boundary_condition(model.velocities.v)

    start_momentum_kernel_args = (model.advection.momentum,
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

    for parameters in kernel_parameters
        launch!(arch, grid, parameters,
                calculate_hydrostatic_free_surface_Gu!, model.timestepper.Gⁿ.u, kernel_offsets, grid, u_kernel_args)
    
        launch!(arch, grid, parameters,
                calculate_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, kernel_offsets, grid, v_kernel_args)
        
        η_parameters = KernelParameters(worktuple(parameters)[1:2], offsets(parameters)[1:2])

        calculate_free_surface_tendency!(grid, model, η_parameters)
    end

    top_tracer_bcs = top_tracer_boundary_conditions(grid, model.tracers)

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))
        @inbounds c_tendency = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection = model.advection[tracer_name]
        @inbounds c_forcing = model.forcing[tracer_name]
        @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])

        tendency_kernel!, closure, diffusivity = tracer_tendency_kernel_function(model, Val(tracer_name), model.closure, model.diffusivity_fields)

        args = tuple(Val(tracer_index),
                     Val(tracer_name),
                     c_advection,
                     closure,
                     c_immersed_bc,
                     model.buoyancy,
                     model.biogeochemistry,
                     model.velocities,
                     model.free_surface,
                     model.tracers,
                     top_tracer_bcs,
                     diffusivity,
                     model.auxiliary_fields,
                     c_forcing,
                     model.clock)

        for parameters in kernel_parameters
            launch!(arch, grid, parameters,
                    tendency_kernel!, c_tendency, kernel_offsets, grid, args)
        end
    end
end

function recompute_auxiliaries!(model, grid, arch)
    
    kernel_parameters = boundary_w_kernel_parameters(grid, arch)

    for parameters in kernel_parameters
        compute_w_from_continuity!(model.velocities, arch, grid; parameters)
    end

    kernel_parameters = boundary_p_kernel_parameters(grid, arch)

    for parameters in kernel_parameters
        update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers; parameters)
    end

    kernel_parameters = boundary_κ_kernel_parameters(grid, model.closure, arch)

    for parameters in kernel_parameters
        calculate_diffusivities!(model.diffusivity_fields, model.closure, model; parameters)
    end
end

# w needs computing in the range - H + 1 : 0 and N - 1 : N + H - 1
function boundary_w_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    Sx  = (Hx, Ny)
    Sy  = (Nx, Hy)
             
    Oᴸx = (-Hx+1, 0)
    Oᴸy = (0, -Hy+1)
    Oᴿx = (Nx-1,  0)
    Oᴿy = (0,  Ny-1)

    sizes = ( Sx,  Sy,  Sx,  Sy)
    offs  = (Oᴸx, Oᴸy, Oᴿx, Oᴿy)
        
    return communicating_boundaries(arch, sizes, offs, grid)
end

# p needs computing in the range  0 : 0 and N + 1 : N + 1
function boundary_p_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)

    Sx  = (1, Ny)
    Sy  = (Nx, 1)
             
    Oᴸx = (-1, 0)
    Oᴸy = (0, -1)
    Oᴿx = (Nx, 0)
    Oᴿy = (0, Ny)

    sizes = ( Sx,  Sy,  Sx,  Sy)
    offs  = (Oᴸx, Oᴸy, Oᴿx, Oᴿy)
        
    return communicating_boundaries(arch, sizes, offs, grid)
end

# diffusivities need computing in the range 0 : B and N - B : N + 1
function boundary_κ_kernel_parameters(grid, closure, arch)
    Nx, Ny, Nz = size(grid)

    B = required_halo_size(closure)

    Sx  = (B+1, Ny, Nz)
    Sy  = (Nx, B+1, Nz)
        
    Oᴸx = (-1, 0, 0)
    Oᴸy = (0, -1, 0)
    Oᴿx = (Nx-B,  0, 0)
    Oᴿy = (0,  Ny-B, 0)

    sizes = ( Sx,  Sy,  Sx,  Sy)
    offs  = (Oᴸx, Oᴸy, Oᴿx, Oᴿy)
        
    return communicating_boundaries(arch, sizes, offs, grid)
end

# tendencies need computing in the range 1 : H and N - H + 1 : N 
function boundary_tendency_kernel_parameters(grid, arch)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, _  = halo_size(grid)
    
    Sx  = (Hx, Ny, Nz)
    Sy  = (Nx, Hy, Nz)
         
    Oᴸx = (0,  0,  0)
    Oᴸy = (0,  0,  0)
    Oᴿx = (Nx-Hx, 0,     0)
    Oᴿy = (0,     Ny-Hy, 0)

    sizes = ( Sx,  Sy,  Sx,  Sy)
    offs  = (Oᴸx, Oᴸy, Oᴿx, Oᴿy)
        
    return communicating_boundaries(arch, sizes, offs, grid)
end

using Oceananigans.Operators: XFlatGrid, YFlatGrid

function communicating_boundaries(arch, S, O, grid) 
    Rx, Ry, _ = arch.ranks

    include_x = !isa(grid, XFlatGrid) && (Rx != 1)
    include_y = !isa(grid, YFlatGrid) && (Ry != 1)

    if include_x && include_y
        return tuple(KernelParameters(S[i], O[i]) for i in 1:4)
    elseif include_x && !(include_y)
        return tuple(KernelParameters(S[i], O[i]) for i in 1:2:3)
    elseif !(include_x) && include_y
        return tuple(KernelParameters(S[i], O[i]) for i in 2:2:4)
    else
        return ()
    end
end

