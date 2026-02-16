using Oceananigans.Biogeochemistry: update_tendencies!
using Oceananigans: fields, TendencyCallsite
using Oceananigans.Models: complete_communication_and_compute_buffer!, interior_tendency_kernel_parameters
using Oceananigans.Grids: get_active_cells_map

import Oceananigans.TimeSteppers: compute_tendencies!
import Oceananigans.TimeSteppers: compute_flux_bc_tendencies!

"""
    compute_tendencies!(model::NonhydrostaticModel, callbacks)

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

    grid = model.grid
    arch = architecture(grid)

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    kernel_parameters = interior_tendency_kernel_parameters(arch, grid)
    active_cells_map  = get_active_cells_map(model.grid, Val(:interior))

    compute_interior_tendency_contributions!(model, kernel_parameters; active_cells_map)
    complete_communication_and_compute_buffer!(model, grid, arch)

    for callback in callbacks
        callback.callsite isa TendencyCallsite && callback(model)
    end

    update_tendencies!(model.biogeochemistry, model)

    return nothing
end

""" Store previous value of the source term and compute current source term. """
function compute_interior_tendency_contributions!(model, kernel_parameters; active_cells_map = nothing)

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
    closure_fields       = model.closure_fields
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
                                closure_fields)

    u_kernel_args = tuple(start_momentum_kernel_args...,
                          u_immersed_bc, end_momentum_kernel_args...,
                          hydrostatic_pressure, clock, forcings.u)

    v_kernel_args = tuple(start_momentum_kernel_args...,
                          v_immersed_bc, end_momentum_kernel_args...,
                          hydrostatic_pressure, clock, forcings.v)

    w_kernel_args = tuple(start_momentum_kernel_args...,
                          w_immersed_bc, end_momentum_kernel_args...,
                          hydrostatic_pressure, clock, forcings.w)

    exclude_periphery = true
    launch!(arch, grid, kernel_parameters, compute_Gu!,
            tendencies.u, grid, u_kernel_args;
            active_cells_map, exclude_periphery)

    launch!(arch, grid, kernel_parameters, compute_Gv!,
            tendencies.v, grid, v_kernel_args;
            active_cells_map, exclude_periphery)

    launch!(arch, grid, kernel_parameters, compute_Gw!,
            tendencies.w, grid, w_kernel_args;
            active_cells_map, exclude_periphery)

    start_tracer_kernel_args = (advection, closure)
    end_tracer_kernel_args   = (buoyancy, biogeochemistry, background_fields, velocities,
                                tracers, auxiliary_fields, closure_fields)

    launch!(arch, grid, kernel_parameters, compute_Gc!,
            values(tendencies)[4:end],
            grid,
            map(i -> Val(i), 1:length(tracers)),
            map(i -> Val(keys(tracers)[i]), 1:length(tracers)),
            start_tracer_kernel_args...,
            map(i -> tracers[i].boundary_conditions.immersed, 1:length(tracers)),
            end_tracer_kernel_args...,
            clock,
            values(forcings)[4:end];
            active_cells_map)

    return nothing
end

#####
##### Tendency calculators for u, v, w-velocity
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function compute_Gu!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function compute_Gv!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the w-velocity equation. """
@kernel function compute_Gw!(Gw, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, grid, args...)
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function compute_Gc!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid, args...)
end

@kernel function compute_Gc!(Gc::Tuple, grid, 
                             val_tracer_index, 
                             val_tracer_name, 
                             advection, 
                             closure,
                             immersed_bc,
                             buoyancy, biogeochemistry, background_fields, velocities,
                             tracers, auxiliary_fields, closure_fields,
                             clock,
                             forcing)
    i, j, k = @index(Global, NTuple)

    N = length(Gc)

    @inbounds for n in 1:N
        Gc[n][i, j, k] = 
            tracer_tendency(i, j, k, grid, 
                            val_tracer_index[n],
                            val_tracer_name[n],
                            advection,
                            closure,
                            immersed_bc[n],
                            buoyancy, biogeochemistry, background_fields, velocities,
                            tracers, auxiliary_fields, closure_fields,
                            clock,
                            forcing[n])
    end
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function compute_flux_bc_tendencies!(model::NonhydrostaticModel)

    Gⁿ    = model.timestepper.Gⁿ
    arch  = model.architecture
    clock = model.clock

    model_fields = fields(model)
    prognostic_fields = merge(model.velocities, model.tracers)

    foreach(i -> compute_x_bcs!(Gⁿ[i], prognostic_fields[i], arch, clock, model_fields), 1:length(prognostic_fields))
    foreach(i -> compute_y_bcs!(Gⁿ[i], prognostic_fields[i], arch, clock, model_fields), 1:length(prognostic_fields))
    foreach(i -> compute_z_bcs!(Gⁿ[i], prognostic_fields[i], arch, clock, model_fields), 1:length(prognostic_fields))

    return nothing
end
