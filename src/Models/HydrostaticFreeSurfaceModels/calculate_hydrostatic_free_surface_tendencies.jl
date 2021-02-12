import Oceananigans.TimeSteppers: calculate_tendencies!

using Oceananigans: fields
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
                                                                        model.forcing,
                                                                        model.clock)

    #=
    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                               model.architecture,
                                               model.velocities,
                                               model.tracers,
                                               model.clock,
                                               fields(model))
    =#

    return nothing
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
                                                                             forcings,
                                                                             clock)

    calculate_Gu_kernel! = calculate_hydrostatic_free_surface_Gu!(device(arch), work_layout(grid, :xyz)...)
    calculate_Gv_kernel! = calculate_hydrostatic_free_surface_Gv!(device(arch), work_layout(grid, :xyz)...)
    calculate_Gc_kernel! = calculate_hydrostatic_free_surface_Gc!(device(arch), work_layout(grid, :xyz)...)
    calculate_Gη_kernel! = calculate_hydrostatic_free_surface_Gη!(device(arch), work_layout(grid, :xy)...)

    barrier = Event(device(arch))

    Gu_event = calculate_Gu_kernel!(tendencies.u, grid, advection.momentum, coriolis, closure,
                                    velocities, free_surface, tracers, diffusivities, hydrostatic_pressure_anomaly,
                                    forcings, clock; dependencies = barrier)

    Gv_event = calculate_Gv_kernel!(tendencies.v, grid, advection.momentum, coriolis, closure,
                                    velocities, free_surface, tracers, diffusivities, hydrostatic_pressure_anomaly,
                                    forcings, clock; dependencies = barrier)
                                    
    Gη_event = calculate_Gη_kernel!(tendencies.η, grid, velocities, free_surface, tracers,
                                    forcings, clock; dependencies = barrier)

    events = [Gu_event, Gv_event, Gη_event]

    for tracer_index in 1:length(tracers)
        @inbounds c_tendency = tendencies[tracer_index+3]
        @inbounds c_advection = advection[tracer_index+1]
        @inbounds forcing = forcings[tracer_index+3]

        Gc_event = calculate_Gc_kernel!(c_tendency, grid, Val(tracer_index),
                                        c_advection, closure, buoyancy,
                                        velocities, free_surface, tracers, diffusivities,
                                        forcing, clock; dependencies = barrier)

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
