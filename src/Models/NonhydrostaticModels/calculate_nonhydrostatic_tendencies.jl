import Oceananigans.TimeSteppers: calculate_tendencies!

using Oceananigans: fields, TimeStepCallsite, TendencyCallsite, UpdateStateCallsite
using Oceananigans.Utils: work_layout, calc_tendency_index

"""
    calculate_tendencies!(model::NonhydrostaticModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model::NonhydrostaticModel, callbacks)

    # Note:
    #
    # "tendencies" is a NamedTuple of OffsetArrays corresponding to the tendency data for use
    # in GPU computations.
    #
    # "model.timestepper.Gⁿ" is a NamedTuple of Fields, whose data also corresponds to
    # tendency data.

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    calculate_interior_tendency_contributions!(model)
                                               
    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                               model.architecture,
                                               model.velocities,
                                               model.tracers,
                                               model.clock,
                                               fields(model))

    [callback(model) for callback in callbacks if isa(callback.callsite, TendencyCallsite)]

    return nothing
end

""" Store previous value of the source term and calculate current source term. """
function calculate_interior_tendency_contributions!(model; dependencies = device_event(model))

    tendencies           = model.timestepper.Gⁿ
    arch                 = model.architecture
    grid                 = model.grid
    advection            = model.advection
    coriolis             = model.coriolis
    buoyancy             = model.buoyancy
    stokes_drift         = model.stokes_drift
    closure              = model.closure
    background_fields    = model.background_fields
    velocities           = model.velocities
    tracers              = model.tracers
    auxiliary_fields     = model.auxiliary_fields
    hydrostatic_pressure = model.pressures.pHY′
    diffusivities        = model.diffusivity_fields
    forcings             = model.forcing
    clock                = model.clock
    u_immersed_bc        = velocities.u.boundary_conditions.immersed
    v_immersed_bc        = velocities.v.boundary_conditions.immersed
    w_immersed_bc        = velocities.w.boundary_conditions.immersed

    start_momentum_kernel_args = (grid,
                                  advection,
                                  coriolis,
                                  stokes_drift,
                                  closure)

    end_momentum_kernel_args = (buoyancy,
                                background_fields,
                                velocities,
                                tracers,
                                auxiliary_fields,
                                diffusivities,
                                forcings)

    u_kernel_args = tuple(start_momentum_kernel_args..., u_immersed_bc, end_momentum_kernel_args..., hydrostatic_pressure, clock)
    v_kernel_args = tuple(start_momentum_kernel_args..., v_immersed_bc, end_momentum_kernel_args..., hydrostatic_pressure, clock)
    w_kernel_args = tuple(start_momentum_kernel_args..., w_immersed_bc, end_momentum_kernel_args..., clock)
    
    only_active_cells = true

    Gu_event = launch!(arch, grid, :xyz, calculate_Gu!, 
                       tendencies.u, u_kernel_args...;
                       dependencies, only_active_cells)

    Gv_event = launch!(arch, grid, :xyz, calculate_Gv!, 
                       tendencies.v, v_kernel_args...;
                       dependencies, only_active_cells)

    Gw_event = launch!(arch, grid, :xyz, calculate_Gw!, 
                       tendencies.w, w_kernel_args...;
                       dependencies, only_active_cells)

    events = [Gu_event, Gv_event, Gw_event]

    start_tracer_kernel_args = (advection, closure)
    end_tracer_kernel_args   = (buoyancy, background_fields, velocities, tracers, auxiliary_fields, diffusivities,
                                forcings, clock)
    
    for tracer_index in 1:length(tracers)
        @inbounds c_tendency = tendencies[tracer_index+3]
        @inbounds forcing = forcings[tracer_index+3]
        @inbounds c_immersed_bc = tracers[tracer_index].boundary_conditions.immersed

        Gc_event = launch!(arch, grid, :xyz, calculate_Gc!,
                           c_tendency, grid, Val(tracer_index),
                           start_tracer_kernel_args..., 
                           c_immersed_bc,
                           end_tracer_kernel_args...;
                           dependencies, only_active_cells)

        push!(events, Gc_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Tendency calculators for u, v, w-velocity
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function calculate_Gu!(Gu, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, args...)
end

@kernel function calculate_Gu!(Gu, grid::ImmersedBoundaryGrid, args...)
    idx = @index(Global, Linear)
    i, j, k = calc_tendency_index(idx, grid)
    @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function calculate_Gv!(Gv, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, args...)
end

@kernel function calculate_Gv!(Gv, grid::ImmersedBoundaryGrid, args...)
    idx = @index(Global, Linear)
    i, j, k = calc_tendency_index(idx, grid)
    @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the w-velocity equation. """
@kernel function calculate_Gw!(Gw, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, args...)
end

@kernel function calculate_Gw!(Gw, grid::ImmersedBoundaryGrid, args...)
    idx = @index(Global, Linear)
    i, j, k = calc_tendency_index(idx, grid)
    @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, grid, args...)
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_Gc!(Gc, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, args...)
end

@kernel function calculate_Gc!(Gc, grid::ImmersedBoundaryGrid, args...)
    idx = @index(Global, Linear)
    i, j, k = calc_tendency_index(idx, 1, 1, 1, grid)
    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid, args...)
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_tendency_contributions!(Gⁿ, arch, velocities, tracers, clock, model_fields)

    barrier = device_event(arch)

    fields = merge(velocities, tracers)

    x_events = Tuple(apply_x_bcs!(Gⁿ[i], fields[i], arch, barrier, clock, model_fields) for i in 1:length(fields))
    y_events = Tuple(apply_y_bcs!(Gⁿ[i], fields[i], arch, barrier, clock, model_fields) for i in 1:length(fields))
    z_events = Tuple(apply_z_bcs!(Gⁿ[i], fields[i], arch, barrier, clock, model_fields) for i in 1:length(fields))
                         
    wait(device(arch), MultiEvent(tuple(x_events..., y_events..., z_events...)))

    return nothing
end