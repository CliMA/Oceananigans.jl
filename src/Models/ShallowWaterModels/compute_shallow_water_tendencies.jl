import Oceananigans.TimeSteppers: compute_tendencies!

using Oceananigans.Utils: launch!
using Oceananigans: fields, TimeStepCallsite, TendencyCallsite, UpdateStateCallsite
using KernelAbstractions: @index, @kernel

using Oceananigans.Architectures: device

using Oceananigans.BoundaryConditions 


"""
    compute_tendencies!(model::ShallowWaterModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function compute_tendencies!(model::ShallowWaterModel, callbacks)

    # Note:
    #
    # "tendencies" is a NamedTuple of OffsetArrays corresponding to the tendency data for use
    # in GPU computations.
    #
    # "model.timestepper.Gⁿ" is a NamedTuple of Fields, whose data also corresponds to
    # tendency data.

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    compute_interior_tendency_contributions!(model.timestepper.Gⁿ,
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
    compute_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                             model.architecture,
                                             model.solution,
                                             model.tracers,
                                             model.clock,
                                             fields(model))

    [callback(model) for callback in callbacks if isa(callback.callsite, TendencyCallsite)]

    return nothing
end

""" Store previous value of the source term and calculate current source term. """
function compute_interior_tendency_contributions!(tendencies,
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

    transport_args = (grid, gravitational_acceleration, advection.momentum, velocities, coriolis, closure, 
                      bathymetry, solution, tracers, diffusivities, forcings, clock, formulation)

    h_args = (grid, gravitational_acceleration, advection.mass, coriolis, closure, 
              solution, tracers, diffusivities, forcings, clock, formulation)

    launch!(arch, grid, :xyz, compute_Guh!, tendencies[1], transport_args...; exclude_periphery=true)
    launch!(arch, grid, :xyz, compute_Gvh!, tendencies[2], transport_args...; exclude_periphery=true)
    launch!(arch, grid, :xyz,  compute_Gh!, tendencies[3], h_args...)

    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        @inbounds Gc = tendencies[tracer_index+3]
        @inbounds forcing = forcings[tracer_index+3]
        @inbounds c_advection = advection[tracer_name]

        launch!(arch, grid, :xyz, compute_Gc!, Gc, grid, Val(tracer_index),
                c_advection, closure, solution, tracers, diffusivities, forcing, clock, formulation)
    end

    return nothing
end

#####
##### Tendency calculators for the transports and height: uh, vh, h
#####

""" Calculate the right-hand-side of the uh-transport equation. """
@kernel function compute_Guh!(Guh,
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
@kernel function compute_Gvh!(Gvh,
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
@kernel function compute_Gh!(Gh,
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
@kernel function compute_Gc!(Gc,
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
function compute_boundary_tendency_contributions!(Gⁿ, arch, solution, tracers, clock, model_fields)
    prognostic_fields = merge(solution, tracers)

    # Solution fields and tracer fields
    for i in 1:length(Gⁿ)
        apply_x_bcs!(Gⁿ[i], prognostic_fields[i], arch, clock, model_fields)
        apply_y_bcs!(Gⁿ[i], prognostic_fields[i], arch, clock, model_fields)
    end

    return nothing
end

