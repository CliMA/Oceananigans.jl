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
                                               model.bathymetry,
                                               model.solution,
                                               model.tracers,
                                               model.diffusivities,
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
                                                    bathymetry,
                                                    solution,
                                                    tracers,
                                                    diffusivities,
                                                    forcings,
                                                    clock)

    barrier = Event(device(arch))

    kernel_arguments = (grid, solution, gravitational_acceleration, advection, coriolis, bathymetry,
                        tracers, diffusivities, forcings, clock)

    workgroup, worksize = work_layout(grid, :xyz)

    # Note that U, V, H may be uh, vh, h (ConservativeSolution) or u, v, η (PrimitiveSolution)
    calculate_GU_kernel! = calculate_GU!(device(arch), workgroup, worksize)
    calculate_GV_kernel! = calculate_GV!(device(arch), workgroup, worksize)
    calculate_GH_kernel! = calculate_GH!(device(arch), workgroup, worksize)

    GU_event = calculate_GU_kernel!(tendencies[1], kernel_arguments...; dependencies=barrier)
    GV_event = calculate_GV_kernel!(tendencies[2], kernel_arguments...; dependencies=barrier)
    GH_event = calculate_GH_kernel!(tendencies[3], kernel_arguments...; dependencies=barrier)

    events = [GU_event, GV_event, GH_event]

    calculate_Gc_kernel! = calculate_Gc!(device(arch), workgroup, worksize)

    for tracer_index in 1:length(tracers)
        @inbounds c_tendency = tendencies[tracer_index+3]
        @inbounds forcing = forcings[tracer_index+3]

        Gc_event = calculate_Gc_kernel!(c_tendency, grid, solution, Val(tracer_index), advection,
                                        tracers, diffusivities, forcing, clock, dependencies=barrier)

        push!(events, Gc_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Tendency calculators for the solution: either uh, vh, h (ConservativeSolution)
##### or u, v, η (PrimitiveSolution)
#####

""" Calculate the right-hand-side of the uh-transport equation. """
@kernel function calculate_GU!(GU, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds GU[i, j, k] = shallow_water_x_momentum_tendency(i, j, k, args...)
end

""" Calculate the right-hand-side of the vh-transport equation. """
@kernel function calculate_GV!(GV, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds GV[i, j, k] = shallow_water_y_momentum_tendency(i, j, k, args...)
end

""" Calculate the right-hand-side of the height equation. """
@kernel function calculate_GH!(GH, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds GH[i, j, k] = shallow_water_height_tendency(i, j, k, args...)
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_Gc!(Gc, args...)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = shallow_water_tracer_tendency(i, j, k, args...)
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

