using Oceananigans.TimeSteppers: calculate_tendencies!, tendency_kernel_size, tendency_kernel_offset
import Oceananigans.TimeSteppers: calculate_tendency_contributions!, calculate_boundary_tendency_contributions!

using Oceananigans.Utils: work_layout
using Oceananigans: fields

using KernelAbstractions: @index, @kernel, Event, MultiEvent

using Oceananigans.Architectures: device
using Oceananigans.BoundaryConditions 

""" Store previous value of the source term and calculate current source term. """
function calculate_tendency_contributions!(model::ShallowWaterModel, region_to_compute; dependencies)

    tendencies    = model.timestepper.Gⁿ
    arch          = model.architecture
    grid          = model.grid
    advection     = model.advection
    velocities    = model.velocities
    coriolis      = model.coriolis
    closure       = model.closure
    bathymetry    = model.bathymetry
    solution      = model.solution
    tracers       = model.tracers
    diffusivities = model.diffusivity_fields
    forcings      = model.forcing
    clock         = model.clock
    formulation   = model.formulation

    gravitational_acceleration = model.gravitational_acceleration
    
    kernel_size = tendency_kernel_size(grid, Val(region_to_compute))[[1, 2]]
    offsets     = tendency_kernel_offset(grid, Val(region_to_compute))[[1, 2]]

    workgroup = heuristic_workgroup(kernel_size...)

    calculate_Guh_kernel! = calculate_Guh!(device(arch), workgroup, kernel_size)
    calculate_Gvh_kernel! = calculate_Gvh!(device(arch), workgroup, kernel_size)
    calculate_Gh_kernel!  =  calculate_Gh!(device(arch), workgroup, kernel_size)
    calculate_Gc_kernel!  =  calculate_Gc!(device(arch), workgroup, kernel_size)

    args_vel = (offsets, grid, gravitational_acceleration, advection.momentum, velocities, coriolis, closure, 
                      bathymetry, solution, tracers, diffusivities, forcings, clock, formulation)
    args_h   = (offsets, grid, gravitational_acceleration, advection.mass, coriolis, closure, 
                      solution, tracers, diffusivities, forcings, clock, formulation)

    Guh_event = calculate_Guh_kernel!(tendencies[1], args_vel...; dependencies)
    Gvh_event = calculate_Gvh_kernel!(tendencies[2], args_vel...; dependencies)
    Gh_event  =  calculate_Gh_kernel!(tendencies[3], args_h...;   dependencies)

    events = [Guh_event, Gvh_event, Gh_event]

    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        @inbounds c_tendency = tendencies[tracer_index+3]
        @inbounds forcing = forcings[tracer_index+3]
        @inbounds c_advection = advection[tracer_name]

        Gc_event = calculate_Gc_kernel!(c_tendency, offsets, grid, Val(tracer_index), c_advection, closure, solution,
                                        tracers, diffusivities, forcing, clock, formulation; dependencies)

        push!(events, Gc_event)
    end

    return events
end

#####
##### Tendency calculators for the transports and height: uh, vh, h
#####

""" Calculate the right-hand-side of the uh-transport equation. """
@kernel function calculate_Guh!(Guh,
                                offsets,
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
    i′ = i + offsets[1]
    j′ = j + offsets[2]
    @inbounds Guh[i′, j′, k] = uh_solution_tendency(i′, j′, k, grid, gravitational_acceleration, advection, velocities, coriolis, closure, 
                                                    bathymetry, solution, tracers, diffusivities, forcings, clock, formulation)
end

""" Calculate the right-hand-side of the vh-transport equation. """
@kernel function calculate_Gvh!(Gvh,
                                offsets,
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
    i′ = i + offsets[1]
    j′ = j + offsets[2]
    @inbounds Gvh[i′, j′, k] = vh_solution_tendency(i′, j′, k, grid, gravitational_acceleration, advection, velocities, coriolis, closure, 
                                                    bathymetry, solution, tracers, diffusivities, forcings, clock, formulation)
end

""" Calculate the right-hand-side of the height equation. """
@kernel function calculate_Gh!(Gh,
                               offsets,
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
    i′ = i + offsets[1]
    j′ = j + offsets[2]
    @inbounds Gh[i′, j′, k] = h_solution_tendency(i′, j′, k, grid, gravitational_acceleration, advection, coriolis, closure,
                                                  solution, tracers, diffusivities, forcings, clock, formulation)
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_Gc!(Gc,
                               offsets,
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
    i′ = i + offsets[1]
    j′ = j + offsets[2]
    @inbounds Gc[i′, j′, k] = tracer_tendency(i′, j′, k, grid, tracer_index, advection, closure, solution, tracers,
                                              diffusivities, forcing, clock, formulation)
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_tendency_contributions!(model::ShallowWaterModel)

    Gⁿ   = model.timestepper.Gⁿ
    grid = model.grid
    arch = model.architecture

    velocities   = model.velocities
    solution     = model.solution
    tracers      = model.tracers
    clock        = model.clock
    model_fields = fields(model)


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

