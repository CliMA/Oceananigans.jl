import Oceananigans.TimeSteppers: calculate_tendencies!

using Oceananigans.Utils: work_layout
using Oceananigans: fields

using KernelAbstractions: @index, @kernel, Event, MultiEvent

using Oceananigans.Architectures: device

using Oceananigans.BoundaryConditions 

"""
    calculate_tendencies!(model::ShallowWaterModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model::ShallowWaterModel)

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
                                               model.gravitational_acceleration,
                                               model.advection,
                                               model.coriolis,
                                               model.closure,
                                               model.bathymetry,
                                               model.solution,
                                               model.tracers,
                                               model.diffusivity_fields,
                                               model.forcing,
                                               model.clock)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                               model.architecture,
                                               model.solution,
                                               model.tracers,
                                               model.clock,
                                               fields(model))

    return nothing
end

""" Store previous value of the source term and calculate current source term. """
function calculate_interior_tendency_contributions!(tendencies,
                                                    arch,
                                                    grid,
                                                    gravitational_acceleration,
                                                    advection,
                                                    coriolis,
                                                    closure, 
                                                    bathymetry,
                                                    solution,
                                                    tracers,
                                                    diffusivities,
                                                    forcings,
                                                    clock)

    workgroup, worksize = work_layout(grid, :xyz)

    calculate_Guh_kernel! = calculate_Guh!(device(arch), workgroup, worksize)
    calculate_Gvh_kernel! = calculate_Gvh!(device(arch), workgroup, worksize)
    calculate_Gh_kernel!  = calculate_Gh!(device(arch), workgroup, worksize)
    calculate_Gc_kernel!  = calculate_Gc!(device(arch), workgroup, worksize)

    barrier = Event(device(arch))

    Guh_event = calculate_Guh_kernel!(tendencies.uh,
                                      grid, gravitational_acceleration, advection, coriolis, closure, 
                                      bathymetry, solution, tracers, diffusivities, forcings, clock,
                                      dependencies=barrier)

    Gvh_event = calculate_Gvh_kernel!(tendencies.vh,
                                      grid, gravitational_acceleration, advection, coriolis, closure, 
                                      bathymetry, solution, tracers, diffusivities, forcings, clock,
                                      dependencies=barrier)

    Gh_event  = calculate_Gh_kernel!(tendencies.h,
                                     grid, gravitational_acceleration, coriolis, closure, 
                                     bathymetry, solution, tracers, diffusivities, forcings, clock,
                                     dependencies=barrier)

    events = [Guh_event, Gvh_event, Gh_event]

    for tracer_index in 1:length(tracers)
        @inbounds c_tendency = tendencies[tracer_index+3]
        @inbounds forcing = forcings[tracer_index+3]

        Gc_event = calculate_Gc_kernel!(c_tendency, grid, Val(tracer_index), advection, closure, solution,
                                        tracers, diffusivities, forcing, clock, dependencies=barrier)

        push!(events, Gc_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Tendency calculators for the transports and height: uh, vh, h
#####

""" Calculate the right-hand-side of the uh-transport equation. """
@kernel function calculate_Guh!(Guh,
                                grid,
                                gravitational_acceleration,
                                advection,
                                coriolis,
                                closure, 
                                bathymetry,
                                solution,
                                tracers,
                                diffusivities,
                                forcings,
                                clock)

    i, j, k = @index(Global, NTuple)

    @inbounds Guh[i, j, k] = uh_solution_tendency(i, j, k, grid, gravitational_acceleration, advection, coriolis, closure, 
                                                    bathymetry, solution, tracers, diffusivities, forcings, clock)
end

""" Calculate the right-hand-side of the vh-transport equation. """
@kernel function calculate_Gvh!(Gvh,
                                grid,
                                gravitational_acceleration,
                                advection,
                                coriolis,
                                closure,
                                bathymetry,
                                solution,
                                tracers,
                                diffusivities,
                                forcings,
                                clock)

    i, j, k = @index(Global, NTuple)

    @inbounds Gvh[i, j, k] = vh_solution_tendency(i, j, k, grid, gravitational_acceleration, advection, coriolis, closure, 
                                                    bathymetry, solution, tracers, diffusivities, forcings, clock)
end

""" Calculate the right-hand-side of the height equation. """
@kernel function calculate_Gh!(Gh,
                               grid,
                               gravitational_acceleration,
                               coriolis,
                               closure,
                               bathymetry,
                               solution,
                               tracers,
                               diffusivities,
                               forcings,
                               clock)

    i, j, k = @index(Global, NTuple)

    @inbounds Gh[i, j, k] = h_solution_tendency(i, j, k, grid, gravitational_acceleration, coriolis, closure, bathymetry,
                                                solution, tracers, diffusivities, forcings, clock)
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
                               solution,
                               tracers,
                               diffusivities,
                               forcing,
                               clock)

    i, j, k = @index(Global, NTuple)

    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid, tracer_index, advection, closure, solution, tracers,
                                            diffusivities, forcing, clock)
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_tendency_contributions!(Gⁿ, arch, solution, tracers, clock, model_fields)

    barrier = Event(device(arch))

    events = []

    # Solution fields
    for i in 1:3
        x_bcs_event = apply_x_bcs!(Gⁿ[i], solution[i], arch, barrier, clock, model_fields)
        y_bcs_event = apply_y_bcs!(Gⁿ[i], solution[i], arch, barrier, clock, model_fields)

        push!(events, x_bcs_event, y_bcs_event)
    end

    # Tracer fields
    for i in 4:length(Gⁿ)
        x_bcs_event = apply_x_bcs!(Gⁿ[i], tracers[i-3], arch, barrier, clock, model_fields)
        y_bcs_event = apply_y_bcs!(Gⁿ[i], tracers[i-3], arch, barrier, clock, model_fields)

        push!(events, x_bcs_event, y_bcs_event)
    end

    events = filter(e -> typeof(e) <: Event, events)
    
    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

