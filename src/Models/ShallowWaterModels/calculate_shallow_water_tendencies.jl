import Oceananigans.TimeSteppers: calculate_tendencies!

using Oceananigans.Utils: work_layout
using Oceananigans: fields, TimestepCallback, TendencyCallback, UpdateStateCallback
using KernelAbstractions: @index, @kernel, Event, MultiEvent

using Oceananigans.Architectures: device

using Oceananigans.BoundaryConditions 


"""
    calculate_tendencies!(model::ShallowWaterModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model::ShallowWaterModel, callbacks)

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
                                               model.velocities,
                                               model.coriolis,
                                               model.closure,
                                               model.bathymetry,
                                               model.solution,
                                               model.tracers,
                                               model.diffusivity_fields,
                                               model.forcing,
                                               model.clock,
                                               model.formulation)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                               model.architecture,
                                               model.solution,
                                               model.tracers,
                                               model.clock,
                                               fields(model))

    [callback(model) for callback in callbacks if isa(callback.callsite, TendencyCallback)]

    return nothing
end

""" Store previous value of the source term and calculate current source term. """
function calculate_interior_tendency_contributions!(tendencies,
                                                    arch,
                                                    grid,
                                                    gravitational_acceleration,
                                                    advection,
                                                    velocities,
                                                    coriolis,
                                                    closure, 
                                                    bathymetry,
                                                    solution,
                                                    tracers,
                                                    diffusivities,
                                                    forcings,
                                                    clock,
                                                    formulation)

    workgroup, worksize = work_layout(grid, :xyz)

    calculate_Guh_kernel! = calculate_Guh!(device(arch), workgroup, worksize)
    calculate_Gvh_kernel! = calculate_Gvh!(device(arch), workgroup, worksize)
    calculate_Gh_kernel!  =  calculate_Gh!(device(arch), workgroup, worksize)
    calculate_Gc_kernel!  =  calculate_Gc!(device(arch), workgroup, worksize)

    barrier = Event(device(arch))

    args_vel = (grid, gravitational_acceleration, advection.momentum, velocities, coriolis, closure, 
                      bathymetry, solution, tracers, diffusivities, forcings, clock, formulation)
    args_h   = (grid, gravitational_acceleration, advection.mass, coriolis, closure, 
                      solution, tracers, diffusivities, forcings, clock, formulation)

    Guh_event = calculate_Guh_kernel!(tendencies[1], args_vel...; dependencies = barrier)
    Gvh_event = calculate_Gvh_kernel!(tendencies[2], args_vel...; dependencies = barrier)
    Gh_event  =  calculate_Gh_kernel!(tendencies[3], args_h...;   dependencies = barrier)

    events = [Guh_event, Gvh_event, Gh_event]

    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        @inbounds c_tendency = tendencies[tracer_index+3]
        @inbounds forcing = forcings[tracer_index+3]
        @inbounds c_advection = advection[tracer_name]

        Gc_event = calculate_Gc_kernel!(c_tendency, grid, Val(tracer_index), c_advection, closure, solution,
                                        tracers, diffusivities, forcing, clock, formulation, dependencies=barrier)

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
                                velocities,
                                coriolis,
                                closure, 
                                bathymetry,
                                solution,
                                tracers,
                                diffusivities,
                                forcings,
                                clock, 
                                formulation)

    i, j, k = @index(Global, NTuple)

    @inbounds Guh[i, j, k] = uh_solution_tendency(i, j, k, grid, gravitational_acceleration, advection, velocities, coriolis, closure, 
                                                    bathymetry, solution, tracers, diffusivities, forcings, clock, formulation)
end

""" Calculate the right-hand-side of the vh-transport equation. """
@kernel function calculate_Gvh!(Gvh,
                                grid,
                                gravitational_acceleration,
                                advection,
                                velocities,
                                coriolis,
                                closure,
                                bathymetry,
                                solution,
                                tracers,
                                diffusivities,
                                forcings,
                                clock, 
                                formulation)

    i, j, k = @index(Global, NTuple)

    @inbounds Gvh[i, j, k] = vh_solution_tendency(i, j, k, grid, gravitational_acceleration, advection, velocities, coriolis, closure, 
                                                    bathymetry, solution, tracers, diffusivities, forcings, clock, formulation)
end

""" Calculate the right-hand-side of the height equation. """
@kernel function calculate_Gh!(Gh,
                               grid,
                               gravitational_acceleration,
                               advection,
                               coriolis,
                               closure,
                               solution,
                               tracers,
                               diffusivities,
                               forcings,
                               clock, 
                               formulation)

    i, j, k = @index(Global, NTuple)

    @inbounds Gh[i, j, k] = h_solution_tendency(i, j, k, grid, gravitational_acceleration, advection, coriolis, closure,
                                                solution, tracers, diffusivities, forcings, clock, formulation)
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
                               clock,
                               formulation)

    i, j, k = @index(Global, NTuple)

    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid, tracer_index, advection, closure, solution, tracers,
                                            diffusivities, forcing, clock, formulation)
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_tendency_contributions!(Gⁿ, arch, solution, tracers, clock, model_fields)

    barrier = Event(device(arch))

    prognostic_fields = merge(solution, tracers)

    events = []

    # Solution fields and tracer fields
    for i in 1:length(Gⁿ)
        x_bcs_event = apply_x_bcs!(Gⁿ[i], prognostic_fields[i], arch, barrier, clock, model_fields)
        y_bcs_event = apply_y_bcs!(Gⁿ[i], prognostic_fields[i], arch, barrier, clock, model_fields)
        push!(events, x_bcs_event, y_bcs_event)
    end

    events = filter(e -> typeof(e) <: Event, events)
    
    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

