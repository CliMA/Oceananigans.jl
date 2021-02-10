import Oceananigans.TimeSteppers: calculate_tendencies!

using Oceananigans: fields
using Oceananigans.Utils: work_layout

"""
    calculate_tendencies!(model::IncompressibleModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model::HydrostaticFreeSurfaceModel)

    # Note:
    #
    # "tendencies" is a NamedTuple of OffsetArrays corresponding to the tendency data for use
    # in GPU computations.
    #
    # "model.timestepper.Gⁿ" is a NamedTuple of Fields, whose data also corresponds to
    # tendency data.

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    calculate_interior_tendency_contributions!(model.timestepper.Gⁿ,
                                               model.architecture,
                                               model.grid,
                                               model.advection,
                                               model.coriolis,
                                               model.buoyancy,
                                               model.closure,
                                               model.velocities,
                                               model.free_surface.η,
                                               model.tracers,
                                               model.pressure.pHY′,
                                               model.diffusivities,
                                               model.forcing,
                                               model.clock)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                               model.architecture,
                                               model.velocities,
                                               model.tracers,
                                               model.clock,
                                               fields(model))

    return nothing
end

""" Store previous value of the source term and calculate current source term. """
function calculate_interior_tendency_contributions!(tendencies,
                                                    arch,
                                                    grid,
                                                    advection,
                                                    coriolis,
                                                    buoyancy,
                                                    closure,
                                                    velocities,
                                                    free_surface_displacement,
                                                    tracers,
                                                    hydrostatic_pressure_anomaly,
                                                    diffusivities,
                                                    forcings,
                                                    clock)

    workgroup, worksize = work_layout(grid, :xyz)
    xy_workgroup, xy_worksize = work_layout(grid, :xy)

    calculate_Gu_kernel! = calculate_Gu!(device(arch), workgroup, worksize)
    calculate_Gv_kernel! = calculate_Gv!(device(arch), workgroup, worksize)
    calculate_Gc_kernel! = calculate_Gc!(device(arch), workgroup, worksize)

    calculate_Gη_kernel! = calculate_Gη!(device(arch), xy_workgroup, xy_worksize)

    barrier = Event(device(arch))

    Gu_event = calculate_Gu_kernel!(tendencies.u, grid, advection, coriolis, closure,
                                    velocities, free_surface_displacement, tracers, diffusivities,
                                    forcings, hydrostatic_pressure_anomaly, clock, dependencies=barrier)

    Gv_event = calculate_Gv_kernel!(tendencies.v, grid, advection, coriolis, closure,
                                    velocities, free_surface_displacement, tracers, diffusivities,
                                    forcings, hydrostatic_pressure_anomaly, clock, dependencies=barrier)

    Gη_event = calculate_Gη_kernel!(tendencies.η, grid, closure,
                                    velocities, free_surface_displacement, tracers, diffusivities,
                                    forcings, clock, dependencies=barrier)

    events = [Gu_event, Gv_event, Gη_event]

    for tracer_index in 1:length(tracers)
        @inbounds c_tendency = tendencies[tracer_index+3]
        @inbounds forcing = forcings[tracer_index+3]

        Gc_event = calculate_Gc_kernel!(c_tendency, grid, Val(tracer_index), advection, closure, buoyancy,
                                        velocities, free_surface_displacement, tracers, diffusivities,
                                        forcing, clock, dependencies=barrier)

        push!(events, Gc_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Tendency calculators for u, v
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function calculate_Gu!(Gu,
                               grid,
                               advection,
                               coriolis,
                               closure,
                               velocities,
                               free_surface_displacement,
                               tracers,
                               diffusivities,
                               forcings,
                               hydrostatic_pressure_anomaly,
                               clock)

    i, j, k = @index(Global, NTuple)

    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, advection, coriolis,
                                                                         closure, velocities, free_surface_displacement, tracers,
                                                                         diffusivities, forcings, hydrostatic_pressure_anomaly, clock)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function calculate_Gv!(Gv,
                               grid,
                               advection,
                               coriolis,
                               closure,
                               velocities,
                               free_surface_displacement,
                               tracers,
                               diffusivities,
                               forcings,
                               hydrostatic_pressure_anomaly,
                               clock)

    i, j, k = @index(Global, NTuple)

    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, advection, coriolis,
                                                                         closure, velocities, free_surface_displacement, tracers,
                                                                         diffusivities, forcings, hydrostatic_pressure_anomaly, clock)
end

""" Calculate the right-hand-side of the w-velocity equation. """
@kernel function calculate_Gη!(Gη,
                               grid,
                               closure,
                               velocities,
                               free_surface_displacement,
                               tracers,
                               diffusivities,
                               forcings,
                               clock)

    i, j = @index(Global, NTuple)

    @inbounds Gη[i, j, 1] = free_surface_tendency(i, j, grid, closure,
                                                  velocities, free_surface_displacement, tracers,
                                                  diffusivities, forcings, clock)
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_Gc!(Gc,
                               grid,
                               tracer_index,
                               advection,
                               closure,
                               buoyancy,
                               velocities,
                               free_surface_displacement,
                               tracers,
                               diffusivities,
                               forcing,
                               clock)

    i, j, k = @index(Global, NTuple)

    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, tracer_index, advection, closure,
                                                                     buoyancy, velocities, free_surface_displacement, tracers,
                                                                     diffusivities, forcing, clock)
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_tendency_contributions!(Gⁿ, arch, velocities, tracers, clock, model_fields)

    barrier = Event(device(arch))

    events = []

    # Velocity fields
    for i in 1:3
        x_bcs_event = apply_x_bcs!(Gⁿ[i], velocities[i], arch, barrier, clock, model_fields)
        y_bcs_event = apply_y_bcs!(Gⁿ[i], velocities[i], arch, barrier, clock, model_fields)
        z_bcs_event = apply_z_bcs!(Gⁿ[i], velocities[i], arch, barrier, clock, model_fields)

        push!(events, x_bcs_event, y_bcs_event, z_bcs_event)
    end

    # Tracer fields
    for i in 4:length(Gⁿ)
        x_bcs_event = apply_x_bcs!(Gⁿ[i], tracers[i-3], arch, barrier, clock, model_fields)
        y_bcs_event = apply_y_bcs!(Gⁿ[i], tracers[i-3], arch, barrier, clock, model_fields)
        z_bcs_event = apply_z_bcs!(Gⁿ[i], tracers[i-3], arch, barrier, clock, model_fields)

        push!(events, x_bcs_event, y_bcs_event, z_bcs_event)
    end

    events = filter(e -> typeof(e) <: Event, events)

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end
