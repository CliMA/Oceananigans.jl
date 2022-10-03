using Oceananigans.TimeSteppers: calculate_tendencies!, tendency_kernel_size, tendency_kernel_offset
import Oceananigans.TimeSteppers: calculate_tendency_contributions!, calculate_boundary_tendency_contributions!

using Oceananigans: fields
using Oceananigans.Utils: work_layout, heuristic_workgroup

""" Store previous value of the source term and calculate current source term. """
function calculate_tendency_contributions!(model::NonhydrostaticModel, region_to_compute; dependencies)

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

    kernel_size = tendency_kernel_size(grid, Val(region_to_compute))
    offsets     = tendency_kernel_offset(grid, Val(region_to_compute))

    workgroup = heuristic_workgroup(kernel_size...)

    calculate_Gu_kernel! = calculate_Gu!(device(arch), workgroup, kernel_size)
    calculate_Gv_kernel! = calculate_Gv!(device(arch), workgroup, kernel_size)
    calculate_Gw_kernel! = calculate_Gw!(device(arch), workgroup, kernel_size)
    calculate_Gc_kernel! = calculate_Gc!(device(arch), workgroup, kernel_size)

    Gu_event = calculate_Gu_kernel!(tendencies.u,
                                    offsets,
                                    grid,
                                    advection,
                                    coriolis,
                                    stokes_drift,
                                    closure,
                                    u_immersed_bc, 
                                    buoyancy,
                                    background_fields,
                                    velocities,
                                    tracers,
                                    auxiliary_fields,
                                    diffusivities,
                                    forcings,
                                    hydrostatic_pressure,
                                    clock;
                                    dependencies)

    Gv_event = calculate_Gv_kernel!(tendencies.v,
                                    offsets,    
                                    grid,
                                    advection,
                                    coriolis,
                                    stokes_drift,
                                    closure,
                                    v_immersed_bc, 
                                    buoyancy,
                                    background_fields,
                                    velocities,
                                    tracers,
                                    auxiliary_fields,
                                    diffusivities,
                                    forcings,
                                    hydrostatic_pressure,
                                    clock;
                                    dependencies)

    Gw_event = calculate_Gw_kernel!(tendencies.w,
                                    offsets,
                                    grid,
                                    advection,
                                    coriolis,
                                    stokes_drift,
                                    closure,
                                    w_immersed_bc, 
                                    buoyancy,
                                    background_fields,
                                    velocities,
                                    tracers,
                                    auxiliary_fields,
                                    diffusivities,
                                    forcings,
                                    clock;
                                    dependencies)

    events = [Gu_event, Gv_event, Gw_event]

    for tracer_index in 1:length(tracers)
        @inbounds c_tendency = tendencies[tracer_index+3]
        @inbounds forcing = forcings[tracer_index+3]
        @inbounds c_immersed_bc = tracers[tracer_index].boundary_conditions.immersed

        Gc_event = calculate_Gc_kernel!(c_tendency,
                                        offsets,
                                        grid,
                                        Val(tracer_index),
                                        advection,
                                        closure,
                                        c_immersed_bc,
                                        buoyancy,
                                        background_fields,
                                        velocities,
                                        tracers,
                                        auxiliary_fields,
                                        diffusivities,
                                        forcing,
                                        clock;
                                        dependencies)

        push!(events, Gc_event)
    end

    return events
end

#####
##### Tendency calculators for u, v, w-velocity
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function calculate_Gu!(Gu, offsets, args...)
    i, j, k = @index(Global, NTuple)
    i′ = i + offsets[1]
    j′ = j + offsets[2]
    k′ = k + offsets[3]
    @inbounds Gu[i′, j′, k′] = u_velocity_tendency(i′, j′, k′, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function calculate_Gv!(Gv, offsets, args...)
    i, j, k = @index(Global, NTuple)
    i′ = i + offsets[1]
    j′ = j + offsets[2]
    k′ = k + offsets[3]
    @inbounds Gv[i′, j′, k′] = v_velocity_tendency(i′, j′, k′, args...)
end

""" Calculate the right-hand-side of the w-velocity equation. """
@kernel function calculate_Gw!(Gw, offsets, args...)
    i, j, k = @index(Global, NTuple)
    i′ = i + offsets[1]
    j′ = j + offsets[2]
    k′ = k + offsets[3]
    @inbounds Gw[i′, j′, k′] = w_velocity_tendency(i′, j′, k′, args...)
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_Gc!(Gc, offsets, args...)
    i, j, k = @index(Global, NTuple)
    i′ = i + offsets[1]
    j′ = j + offsets[2]
    k′ = k + offsets[3]
    @inbounds Gc[i′, j′, k′] = tracer_tendency(i′, j′, k′, args...)
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_tendency_contributions!(model::NonhydrostaticModel)
           
    Gⁿ = model.timestepper.Gⁿ

    arch  = model.architecture
    clock = model.clock

    model_fields = fields(model)

    barrier = device_event(arch)

    prognostic_fields = merge(model.velocities, model.tracers)

    x_events = Tuple(apply_x_bcs!(Gⁿ[i], prognostic_fields[i], arch, barrier, clock, model_fields) for i in 1:length(prognostic_fields))
    y_events = Tuple(apply_y_bcs!(Gⁿ[i], prognostic_fields[i], arch, barrier, clock, model_fields) for i in 1:length(prognostic_fields))
    z_events = Tuple(apply_z_bcs!(Gⁿ[i], prognostic_fields[i], arch, barrier, clock, model_fields) for i in 1:length(prognostic_fields))
                         
    wait(device(arch), MultiEvent(tuple(x_events..., y_events..., z_events...)))

    return nothing
end
