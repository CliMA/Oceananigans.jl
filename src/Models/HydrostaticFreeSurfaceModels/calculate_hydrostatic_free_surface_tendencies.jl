import Oceananigans.TimeSteppers: calculate_tendencies!
import Oceananigans: tracer_tendency_kernel_function

using Oceananigans.Architectures: device_event
using Oceananigans: fields, prognostic_fields, TimeStepCallsite, TendencyCallsite, UpdateStateCallsite
using Oceananigans.Utils: work_layout, calc_tendency_index
using Oceananigans.Fields: immersed_boundary_condition

using Oceananigans.ImmersedBoundaries: use_only_active_cells, ActiveCellsIBG

"""
    calculate_tendencies!(model::HydrostaticFreeSurfaceModel, callbacks)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model::HydrostaticFreeSurfaceModel, callbacks)

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    calculate_hydrostatic_free_surface_interior_tendency_contributions!(model)
    calculate_hydrostatic_free_surface_advection_tendency_contributions!(model)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_hydrostatic_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                                           model.grid,
                                                           model.architecture,
                                                           model.velocities,
                                                           model.free_surface,
                                                           model.tracers,
                                                           model.clock,
                                                           fields(model),
                                                           model.closure,
                                                           model.buoyancy)

    [callback(model) for callback in callbacks if isa(callback.callsite, TendencyCallsite)]

    return nothing
end

function calculate_free_surface_tendency!(grid, model, dependencies)

    arch = architecture(grid)

    Gη_event = launch!(arch, grid, :xy,
                       calculate_hydrostatic_free_surface_Gη!, model.timestepper.Gⁿ.η,
                       grid,
                       model.velocities,
                       model.free_surface,
                       model.tracers,
                       model.auxiliary_fields,
                       model.forcing,
                       model.clock;
                       dependencies = dependencies)

    return Gη_event
end
    

""" Calculate momentum tendencies if momentum is not prescribed. `velocities` argument eases dispatch on `PrescribedVelocityFields`."""
function calculate_hydrostatic_momentum_tendencies!(model, velocities; dependencies = device_event(model))

    grid = model.grid
    arch = architecture(grid)

    u_immersed_bc = immersed_boundary_condition(velocities.u)
    v_immersed_bc = immersed_boundary_condition(velocities.v)

    start_momentum_kernel_args = (grid,
                                  model.advection.momentum,
                                  model.coriolis,
                                  model.closure)

    end_momentum_kernel_args = (velocities,
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

    Gu_event = launch!(arch, grid, :xyz,
                       calculate_hydrostatic_free_surface_Gu!, model.timestepper.Gⁿ.u, u_kernel_args...;
                       dependencies = dependencies, only_active_cells)

    Gv_event = launch!(arch, grid, :xyz,
                       calculate_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, v_kernel_args...;
                       dependencies = dependencies, only_active_cells)

    Gη_event = calculate_free_surface_tendency!(grid, model, dependencies)

    events = [Gu_event, Gv_event, Gη_event]

    return events
end

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVD, CATKEVDArray

# Fallback
@inline tracer_tendency_kernel_function(model::HydrostaticFreeSurfaceModel, closure, tracer_name) =
    hydrostatic_free_surface_tracer_tendency

@inline tracer_tendency_kernel_function(model::HydrostaticFreeSurfaceModel, closure::CATKEVD, ::Val{:e}) =
    hydrostatic_turbulent_kinetic_energy_tendency

function tracer_tendency_kernel_function(model::HydrostaticFreeSurfaceModel, closures::Tuple, ::Val{:e})
    if any(cl isa Union{CATKEVD, CATKEVDArray} for cl in closures)
        return hydrostatic_turbulent_kinetic_energy_tendency
    else
        return hydrostatic_free_surface_tracer_tendency
    end
end

top_tracer_boundary_conditions(grid, tracers) =
    NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

""" Store previous value of the source term and calculate current source term. """
function calculate_hydrostatic_free_surface_interior_tendency_contributions!(model)

    arch = model.architecture
    grid = model.grid

    barrier = device_event(model)

    events = calculate_hydrostatic_momentum_tendencies!(model, model.velocities; dependencies = barrier)

    top_tracer_bcs = top_tracer_boundary_conditions(grid, model.tracers)

    only_active_cells = use_only_active_cells(grid)

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))
        @inbounds c_tendency = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection = model.advection[tracer_name]
        @inbounds c_forcing = model.forcing[tracer_name]
        @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])
        c_kernel_function = tracer_tendency_kernel_function(model, model.closure, Val(tracer_name))

        Gc_event = launch!(arch, grid, :xyz,
                           calculate_hydrostatic_free_surface_Gc!,
                           c_tendency,
                           c_kernel_function,
                           grid,
                           Val(tracer_index),
                           c_advection,
                           model.closure,
                           c_immersed_bc,
                           model.buoyancy,
                           model.velocities,
                           model.free_surface,
                           model.tracers,
                           top_tracer_bcs,
                           model.diffusivity_fields,
                           model.auxiliary_fields,
                           c_forcing,
                           model.clock;
                           dependencies = barrier, 
                           only_active_cells)

        push!(events, Gc_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

""" Calculate advection of prognostic quantities. """
function calculate_hydrostatic_free_surface_advection_tendency_contributions!(model)

    arch = model.architecture
    grid = model.grid
    Nx, Ny, Nz = N = size(grid)

    barrier = device_event(model)
    Ix, Iy, Iz = 2, 2, 2
    for t in [2, 3, 4, 5, 6, 7, 8]
        if mod(Nx, t) == 0
            Ix = t
        end
        if mod(Ny, t) == 0 
            Iy = t
        end
        if mod(Nz, t) == 0
            Iz = t
        end
    end

    workgroup = (min(Ix, Nx),  min(Iy, Ny),  min(Iz, Nz))
    worksize  = N

    array_size = arch_array(arch, [halo_size(grid)...])
    halo       = arch_array(arch, [min.(size(grid), halo_size(grid))...])

    advection_contribution! = _calculate_hydrostatic_free_surface_advection!(Architectures.device(arch), workgroup, worksize)
    advection_event         = advection_contribution!(model.timestepper.Gⁿ,
                                                      grid,
                                                      model.advection,
                                                      model.velocities,
                                                      model.tracers,
                                                      array_size,
                                                      halo;
                                                      dependencies = barrier)
    
    wait(device(arch), advection_event)

    return nothing
end

#####
##### Tendency calculators for u, v
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function calculate_hydrostatic_free_surface_Gu!(Gu, grid, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

@kernel function calculate_hydrostatic_free_surface_Gu!(Gu, grid::ActiveCellsIBG, args...)
    idx = @index(Global, Linear)
    i, j, k = calc_tendency_index(idx, grid)
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function calculate_hydrostatic_free_surface_Gv!(Gv, grid, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

@kernel function calculate_hydrostatic_free_surface_Gv!(Gv, grid::ActiveCellsIBG, args...)
    idx = @index(Global, Linear)
    i, j, k = calc_tendency_index(idx, grid)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

using Oceananigans.Grids: halo_size

@inline @propagate_inbounds Base.getindex(v::DisplacedSharedArray, i, j, k)       = v.s_array[i + v.i, j + v.j, k + v.k]
@inline @propagate_inbounds Base.setindex!(v::DisplacedSharedArray, val, i, j, k) = setindex!(v.s_array, val, i + v.i, j + v.j, k + v.k)

@inline @propagate_inbounds Base.lastindex(v::DisplacedSharedArray)      = lastindex(v.s_array)
@inline @propagate_inbounds Base.lastindex(v::DisplacedSharedArray, dim) = lastindex(v.s_array, dim)

@kernel function _calculate_hydrostatic_free_surface_advection!(Gⁿ, grid::AbstractGrid{FT}, advection, velocities, tracers, array_size, halo) where FT
    i,  j,  k  = @index(Global, NTuple)
    is, js, ks = @index(Local,  NTuple)

    N = @uniform @groupsize()[1]
    M = @uniform @groupsize()[2]
    O = @uniform @groupsize()[3]

    @synchronize

    us = @localmem FT (N+2*array_size[1], M+2*array_size[2], O+2*array_size[3])
    vs = @localmem FT (N+2*array_size[1], M+2*array_size[2], O+2*array_size[3])
    ws = @localmem FT (N+2*array_size[1], M+2*array_size[2], O+2*array_size[3])
    cs = @localmem FT (N+2*array_size[1], M+2*array_size[2], O+2*array_size[3])

    @inbounds us[is+halo[1], js+halo[2], k+halo[3]] = velocities.u[i, j, k]
    @inbounds vs[is+halo[1], js+halo[2], k+halo[3]] = velocities.v[i, j, k]
    @inbounds ws[is+halo[1], js+halo[2], k+halo[3]] = velocities.w[i, j, k]

    if is <= halo[1]
        @inbounds us[is, js+halo[2], ks+halo[3]] = velocities.u[i - halo[1], j, k]
        @inbounds vs[is, js+halo[2], ks+halo[3]] = velocities.v[i - halo[1], j, k]
        @inbounds ws[is, js+halo[2], ks+halo[3]] = velocities.w[i - halo[1], j, k]
    end
    if is >= N - halo[1] + 1
        @inbounds us[is+2halo[1], js+halo[2], ks+halo[3]] = velocities.u[i + halo[1], j, k]
        @inbounds vs[is+2halo[1], js+halo[2], ks+halo[3]] = velocities.v[i + halo[1], j, k]
        @inbounds ws[is+2halo[1], js+halo[2], ks+halo[3]] = velocities.w[i + halo[1], j, k]
        # Fill the angles because of staggering!
        if js <= halo[2]
            @inbounds us[is+2halo[1], js, ks+halo[3]] = velocities.u[i + halo[1], j - halo[2], k]
        end
        if js >= M - halo[2] + 1
            @inbounds us[is+2halo[1], js+2halo[2], ks+halo[3]] = velocities.u[i + halo[1], j + halo[2], k]
        end
        if ks <= halo[3]
            @inbounds us[is+2halo[1], js+halo[2], ks] = velocities.u[i + halo[1], j, k - halo[3]]
        end
        if ks >= O - halo[3] + 1    
            @inbounds us[is+2halo[1], js+halo[2], ks+2halo[3]] = velocities.u[i + halo[1], j, k + halo[3]]
        end
    end

    if js <= halo[2]
        @inbounds us[is+halo[1],js, ks+halo[3]] = velocities.u[i, j - halo[2], k]
        @inbounds vs[is+halo[1],js, ks+halo[3]] = velocities.v[i, j - halo[2], k]
        @inbounds ws[is+halo[1],js, ks+halo[3]] = velocities.w[i, j - halo[2], k]
    end
    if js >= M - halo[2] + 1
        @inbounds us[is+halo[1], js+2halo[2], k+halo[3]] = velocities.u[i, j + halo[2], k]
        @inbounds vs[is+halo[1], js+2halo[2], k+halo[3]] = velocities.v[i, j + halo[2], k]
        @inbounds ws[is+halo[1], js+2halo[2], k+halo[3]] = velocities.w[i, j + halo[2], k]
        # Fill the angles because of staggering!
        if is <= halo[1]
            @inbounds vs[is, js+2halo[2], ks+halo[3]] = velocities.v[i - halo[1], j + halo[2], k]
        end
        if is >= N - halo[1] + 1
            @inbounds vs[is+2halo[1], js+2halo[2], k+halo[3]] = velocities.v[i + halo[1], j + halo[2], k]
        end
        if ks <= halo[3]
            @inbounds vs[is+halo[1], js+2halo[2], ks] = velocities.v[i, j - halo[2], k - halo[3]]
        end
        if ks >= O - halo[3] + 1
            @inbounds vs[is+halo[1], js+2halo[2], ks+2halo[3]] = velocities.v[i, j + halo[2], k + halo[3]]
        end
    end
    
    if ks <= halo[3]
        @inbounds us[is+halo[1], js+halo[2], ks] = velocities.u[i, j, k - halo[3]]
        @inbounds vs[is+halo[1], js+halo[2], ks] = velocities.v[i, j, k - halo[3]]
        @inbounds ws[is+halo[1], js+halo[2], ks] = velocities.w[i, j, k - halo[3]]
    end
    if ks >= O - halo[3] + 1
        @inbounds us[is+halo[1], js+halo[2], ks+2halo[3]] = velocities.u[i, j, k + halo[3]]
        @inbounds vs[is+halo[1], js+halo[2], ks+2halo[3]] = velocities.v[i, j, k + halo[3]]
        @inbounds ws[is+halo[1], js+halo[2], ks+2halo[3]] = velocities.w[i, j, k + halo[3]]
        # Fill the angles because of staggering!
        if is <= halo[1]
            @inbounds ws[is, js+halo[2], ks+2halo[3]] = velocities.w[i - halo[1], j, k + halo[3]]
        end
        if is >= N - halo[1] + 1
            @inbounds ws[is+2halo[1], js+halo[2], ks+2halo[3]] = velocities.w[i + halo[1], j, k + halo[3]]
        end
        if js <= halo[2]
            @inbounds ws[is+halo[1], js, ks+2halo[3]] = velocities.w[i, j - halo[2], k + halo[3]]
        end
        if js >= M - halo[2] + 1
            @inbounds ws[is+halo[1], js+2halo[2], ks+2halo[3]] = velocities.w[i, j + halo[2], k + halo[3]]
        end
    end

    @synchronize

    @inbounds Gⁿ.u[i, j, k] -= U_dot_∇u(i, j, k, grid, advection.momentum, (u = us, v = vs, w = ws), is+halo[1], js+halo[2], ks+halo[3])
    @inbounds Gⁿ.v[i, j, k] -= U_dot_∇v(i, j, k, grid, advection.momentum, (u = us, v = vs, w = ws), is+halo[1], js+halo[2], ks+halo[3])

    ntuple(Val(length(tracers))) do n
        Base.@_inline_meta
        tracer = tracers[n]
        @inbounds cs[is+halo[1], js+halo[2], ks+halo[3]] = tracer[i, j, k]
    
        # No corners needed for the tracer
        if is <= halo[1]
            @inbounds cs[is, js+halo[2], ks+halo[3]] = tracer[i - halo[1], j, k]
        end
        if is >= N - halo[1] + 1
            @inbounds cs[is+2halo[1], js+halo[2], ks+halo[3]] = tracer[i + halo[1], j, k]
        end
    
        if js <= halo[2]
            @inbounds cs[is+halo[1], js, ks+halo[3]] = tracer[i, j - halo[2], k]
        end
        if js >= M - halo[2] + 1
            @inbounds cs[is+halo[1], js+2halo[2], ks+halo[3]] = tracer[i, j + halo[2], k]
        end
        
        if ks <= halo[3]
            @inbounds cs[is+halo[1], js+halo[2], ks] = tracer[i, j, k - halo[3]]
        end
        if ks >= O - halo[3] + 1
            @inbounds cs[is+halo[1], j+halo[2], k+2halo[3]] = tracer[i, j, k + halo[3]]
        end
    
        @synchronize

        @inbounds Gⁿ[n+3][i, j, k] -= div_Uc(i, j, k, grid, advection[n+1], (u = us, v = vs, w = ws), cs, is+halo[1], js+halo[2], ks+halo[3])
    end
end

#####
##### Tendency calculators for tracers
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_hydrostatic_free_surface_Gc!(Gc, tendency_kernel_function, grid, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = tendency_kernel_function(i, j, k, grid, args...)
end

@kernel function calculate_hydrostatic_free_surface_Gc!(Gc, tendency_kernel_function, grid::ActiveCellsIBG, args...)
    idx = @index(Global, Linear)
    i, j, k = calc_tendency_index(idx, grid)
    @inbounds Gc[i, j, k] = tendency_kernel_function(i, j, k, grid, args...)
end

#####
##### Tendency calculators for an explicit free surface
#####

""" Calculate the right-hand-side of the free surface displacement (``η``) equation. """
@kernel function calculate_hydrostatic_free_surface_Gη!(Gη, grid, args...)
    i, j = @index(Global, NTuple)
    @inbounds Gη[i, j, grid.Nz+1] = free_surface_tendency(i, j, grid, args...)
end

#####
##### Boundary condributions to hydrostatic free surface model
#####

function apply_flux_bcs!(Gcⁿ, events, c, arch, barrier, args...)
    x_bcs_event = apply_x_bcs!(Gcⁿ, c, arch, barrier, args...)
    y_bcs_event = apply_y_bcs!(Gcⁿ, c, arch, barrier, args...)
    z_bcs_event = apply_z_bcs!(Gcⁿ, c, arch, barrier, args...)

    push!(events, x_bcs_event, y_bcs_event, z_bcs_event)

    return nothing
end

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_hydrostatic_boundary_tendency_contributions!(Gⁿ, grid, arch, velocities, free_surface, tracers, args...)

    barrier = device_event(arch)

    events = []

    # Velocity fields
    for i in (:u, :v)
        apply_flux_bcs!(Gⁿ[i], events, velocities[i], arch, barrier, args...)
    end

    # Free surface
    apply_flux_bcs!(Gⁿ.η, events, displacement(free_surface), arch, barrier, args...)

    # Tracer fields
    for i in propertynames(tracers)
        apply_flux_bcs!(Gⁿ[i], events, tracers[i], arch, barrier, args...)
    end

    events = filter(e -> typeof(e) <: Event, events)

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end
