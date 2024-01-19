using Oceananigans.Biogeochemistry: update_tendencies!
using Oceananigans: fields, TendencyCallsite
using Oceananigans.Utils: work_layout
using Oceananigans.Models: complete_communication_and_compute_boundary!, interior_tendency_kernel_parameters

using Oceananigans.ImmersedBoundaries: use_only_active_interior_cells, ActiveCellsIBG, 
                                       InteriorMap, active_linear_index_to_tuple

import Oceananigans.TimeSteppers: compute_tendencies!

"""
    compute_tendencies!(model::NonhydrostaticModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function compute_tendencies!(model::NonhydrostaticModel, callbacks)

    # Note:
    #
    # "tendencies" is a NamedTuple of OffsetArrays corresponding to the tendency data for use
    # in GPU computations.
    #
    # "model.timestepper.Gⁿ" is a NamedTuple of Fields, whose data also corresponds to
    # tendency data.

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    kernel_parameters = tuple(interior_tendency_kernel_parameters(model.grid))

    compute_interior_tendency_contributions!(model, kernel_parameters; only_active_cells = use_only_active_interior_cells(model.grid))
    complete_communication_and_compute_boundary!(model, model.grid, model.architecture)
                      
    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    compute_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                               model.architecture,
                                               model.velocities,
                                               model.tracers,
                                               model.clock,
                                               fields(model))

    for callback in callbacks
        callback.callsite isa TendencyCallsite && callback(model)
    end

    update_tendencies!(model.biogeochemistry, model)

    return nothing
end

""" Store previous value of the source term and compute current source term. """
function compute_interior_tendency_contributions!(model, kernel_parameters; only_active_cells = nothing)

    tendencies           = model.timestepper.Gⁿ
    arch                 = model.architecture
    grid                 = model.grid
    advection            = model.advection
    coriolis             = model.coriolis
    buoyancy             = model.buoyancy
    biogeochemistry      = model.biogeochemistry
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

    start_momentum_kernel_args = (advection,
                                  coriolis,
                                  stokes_drift,
                                  closure)

    end_momentum_kernel_args = (buoyancy,
                                background_fields,
                                velocities,
                                tracers,
                                auxiliary_fields,
                                diffusivities)

    u_kernel_args = tuple(start_momentum_kernel_args..., u_immersed_bc, end_momentum_kernel_args..., forcings, hydrostatic_pressure, clock)
    v_kernel_args = tuple(start_momentum_kernel_args..., v_immersed_bc, end_momentum_kernel_args..., forcings, hydrostatic_pressure, clock)
    w_kernel_args = tuple(start_momentum_kernel_args..., w_immersed_bc, end_momentum_kernel_args..., forcings, clock)

    for parameters in kernel_parameters
        launch!(arch, grid, parameters, compute_Gu!, 
                tendencies.u, grid, only_active_cells, u_kernel_args;
                only_active_cells)

        launch!(arch, grid, parameters, compute_Gv!, 
                tendencies.v, grid, only_active_cells, v_kernel_args;
                only_active_cells)

        launch!(arch, grid, parameters, compute_Gw!, 
                tendencies.w, grid, only_active_cells, w_kernel_args;
                only_active_cells)
    end

    start_tracer_kernel_args = (advection, closure)
    end_tracer_kernel_args   = (buoyancy, biogeochemistry, background_fields, velocities, tracers, auxiliary_fields, diffusivities)

    for tracer_index in 1:length(tracers)
        @inbounds c_tendency = tendencies[tracer_index + 3]
        @inbounds forcing = forcings[tracer_index + 3]
        @inbounds c_immersed_bc = tracers[tracer_index].boundary_conditions.immersed
        @inbounds tracer_name = keys(tracers)[tracer_index]

        args = tuple(Val(tracer_index), Val(tracer_name),
                     start_tracer_kernel_args..., 
                     c_immersed_bc,
                     end_tracer_kernel_args...,
                     forcing, clock)

        for parameters in kernel_parameters
            launch!(arch, grid, parameters, compute_Gc!, 
                    c_tendency, grid, only_active_cells, args;
                    only_active_cells)
        end
    end

    return nothing
end

#####
##### Tendency calculators for u, v, w-velocity
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function compute_Gu!(Gu, grid, interior_map, args) 
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, grid, args...)
end

@kernel function compute_Gu!(Gu, grid::ActiveCellsIBG, map::InteriorMap, args) 
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_tuple(idx, map, grid)
    @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function compute_Gv!(Gv, grid, interior_map, args) 
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, grid, args...)
end

@kernel function compute_Gv!(Gv, grid::ActiveCellsIBG, map::InteriorMap, args) 
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_tuple(idx, map, grid)
    @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the w-velocity equation. """
@kernel function compute_Gw!(Gw, grid, interior_map, args) 
    i, j, k = @index(Global, NTuple)
    @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, grid, args...)
end

@kernel function compute_Gw!(Gw, grid::ActiveCellsIBG, map, ::InteriorMap, args)
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_tuple(idx, map, grid)
    @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, grid, args...)
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function compute_Gc!(Gc, grid, interior_map, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid, args...)
end

@kernel function compute_Gc!(Gc, grid::ActiveCellsIBG, map::InteriorMap, args) 
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_tuple(idx, map, grid)
    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid, args...)
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function compute_boundary_tendency_contributions!(Gⁿ, arch, velocities, tracers, clock, model_fields)
    fields = merge(velocities, tracers)

    foreach(i -> apply_x_bcs!(Gⁿ[i], fields[i], arch, clock, model_fields), 1:length(fields))
    foreach(i -> apply_y_bcs!(Gⁿ[i], fields[i], arch, clock, model_fields), 1:length(fields))
    foreach(i -> apply_z_bcs!(Gⁿ[i], fields[i], arch, clock, model_fields), 1:length(fields))
                         
    return nothing
end
