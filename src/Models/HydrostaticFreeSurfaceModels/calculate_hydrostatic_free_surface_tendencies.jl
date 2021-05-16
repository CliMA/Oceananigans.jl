import Oceananigans.TimeSteppers: calculate_tendencies!

using Oceananigans: fields, prognostic_fields
using Oceananigans.Utils: work_layout

"""
    calculate_tendencies!(model::IncompressibleModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model::HydrostaticFreeSurfaceModel)

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    calculate_hydrostatic_free_surface_interior_tendency_contributions!(model.timestepper.Gⁿ,
                                                                        model.architecture,
                                                                        model.grid,
                                                                        model.advection,
                                                                        model.coriolis,
                                                                        model.buoyancy,
                                                                        model.closure,
                                                                        model.velocities,
                                                                        model.free_surface,
                                                                        model.tracers,
                                                                        model.pressure.pHY′,
                                                                        model.diffusivities,
                                                                        model.auxiliary_fields,
                                                                        model.forcing,
                                                                        model.clock)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_hydrostatic_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                                           model.architecture,
                                                           model.velocities,
                                                           model.free_surface,
                                                           model.tracers,
                                                           model.clock,
                                                           fields(model))

    return nothing
end

function calculate_hydrostatic_momentum_tendencies!(tendencies, velocities, arch, grid, advection, coriolis, closure,
                                                    free_surface, tracers, diffusivities, hydrostatic_pressure_anomaly,
                                                    auxiliary_fields, forcings, clock, barrier)

    Gu_event = launch!(arch, grid, :xyz, calculate_hydrostatic_free_surface_Gu!,
                       tendencies.u, grid, advection.momentum, coriolis, closure,
                       velocities, free_surface, tracers, diffusivities, hydrostatic_pressure_anomaly,
                       auxiliary_fields, forcings, clock; dependencies = barrier)

    Gv_event = launch!(arch, grid, :xyz, calculate_hydrostatic_free_surface_Gv!,
                       tendencies.v, grid, advection.momentum, coriolis, closure,
                       velocities, free_surface, tracers, diffusivities, hydrostatic_pressure_anomaly,
                       auxiliary_fields, forcings, clock; dependencies = barrier)

    Gη_event = launch!(arch, grid, :xy, calculate_hydrostatic_free_surface_Gη!,
                       tendencies.η, grid, velocities, free_surface, tracers,
                       auxiliary_fields, forcings, clock; dependencies = barrier)

    events = [Gu_event, Gv_event, Gη_event]

    return events
end

""" Store previous value of the source term and calculate current source term. """
function calculate_hydrostatic_free_surface_interior_tendency_contributions!(tendencies,
                                                                             arch,
                                                                             grid,
                                                                             advection,
                                                                             coriolis,
                                                                             buoyancy,
                                                                             closure,
                                                                             velocities,
                                                                             free_surface,
                                                                             tracers,
                                                                             hydrostatic_pressure_anomaly,
                                                                             diffusivities,
                                                                             auxiliary_fields,
                                                                             forcings,
                                                                             clock)

    barrier = Event(device(arch))

    events = calculate_hydrostatic_momentum_tendencies!(tendencies, velocities, arch, grid, advection, coriolis, closure,
                                                        free_surface, tracers, diffusivities, hydrostatic_pressure_anomaly,
                                                        auxiliary_fields, forcings, clock, barrier)

    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        @inbounds c_tendency = tendencies[tracer_name]
        @inbounds c_advection = advection[tracer_name]
        @inbounds forcing = forcings[tracer_name]

        Gc_event = launch!(arch, grid, :xyz, calculate_hydrostatic_free_surface_Gc!,
                           c_tendency, grid, Val(tracer_index),
                           c_advection, closure, buoyancy,
                           velocities, free_surface, tracers, diffusivities,
                           auxiliary_fields, forcing, clock;
                           dependencies=barrier)

        push!(events, Gc_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

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

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function calculate_hydrostatic_free_surface_Gv!(Gv, grid, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for tracers
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_hydrostatic_free_surface_Gc!(Gc, grid, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for an explicit free surface
#####

""" Calculate the right-hand-side of the free surface displacement (η) equation. """
@kernel function calculate_hydrostatic_free_surface_Gη!(Gη, grid, args...)
    i, j = @index(Global, NTuple)
    @inbounds Gη[i, j, 1] = free_surface_tendency(i, j, grid, args...)
end

#####
##### Boundary condributions to hydrostatic free surface model
#####

function apply_flux_bcs!(Gcⁿ, events, c, arch, barrier, clock, model_fields)
    x_bcs_event = apply_x_bcs!(Gcⁿ, c, arch, barrier, clock, model_fields)
    y_bcs_event = apply_y_bcs!(Gcⁿ, c, arch, barrier, clock, model_fields)
    z_bcs_event = apply_z_bcs!(Gcⁿ, c, arch, barrier, clock, model_fields)

    push!(events, x_bcs_event, y_bcs_event, z_bcs_event)

    return nothing
end

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_hydrostatic_boundary_tendency_contributions!(Gⁿ, arch, velocities, free_surface, tracers, clock, model_fields)

    barrier = Event(device(arch))

    events = []

    # Velocity fields
    for i in (:u, :v)
        apply_flux_bcs!(Gⁿ[i], events, velocities[i], arch, barrier, clock, model_fields)
    end

    # Free surface
    apply_flux_bcs!(Gⁿ.η, events, displacement(free_surface), arch, barrier, clock, model_fields)

    # Tracer fields
    for i in propertynames(tracers)
        apply_flux_bcs!(Gⁿ[i], events, tracers[i], arch, barrier, clock, model_fields)
    end

    events = filter(e -> typeof(e) <: Event, events)

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end
