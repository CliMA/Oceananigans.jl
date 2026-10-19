using Oceananigans: fields, prognostic_fields, TendencyCallsite
using Oceananigans.Biogeochemistry: update_tendencies!, add_biogeochemical_transitions!
using Oceananigans.Models: complete_communication_and_compute_buffer!, interior_tendency_kernel_parameters

"""
$(TYPEDSIGNATURES)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function Oceananigans.TimeSteppers.compute_tendencies!(model::NonhydrostaticModel, callbacks)

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

    compute_interior_tendency_contributions!(model, kernel_parameters)
    complete_communication_and_compute_buffer!(model, grid, arch)
    compute_biogeochemical_transitions!(model, :xyz)

    for callback in callbacks
        callback.callsite isa TendencyCallsite && callback(model)
    end

    update_tendencies!(model.biogeochemistry, model)

    return nothing
end

""" Store previous value of the source term and compute current source term. """
function compute_interior_tendency_contributions!(model, kernel_parameters)

    tendencies           = model.timestepper.Gⁿ
    arch                 = model.architecture
    grid                 = model.grid
    advection            = model.advection.momentum
    coriolis             = model.coriolis
    buoyancy             = model.buoyancy
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
    u_bcs                = NamedTuple(velocities.u.boundary_conditions)
    v_bcs                = NamedTuple(velocities.v.boundary_conditions)
    w_bcs                = NamedTuple(velocities.w.boundary_conditions)

    exclude_periphery = true
    launch!(arch, grid, kernel_parameters, compute_Gu!,
            tendencies.u, grid,
            advection, coriolis, stokes_drift, closure, u_bcs, buoyancy, background_fields,
            velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcings.u;
            exclude_periphery)

    launch!(arch, grid, kernel_parameters, compute_Gv!,
            tendencies.v, grid,
            advection, coriolis, stokes_drift, closure, v_bcs, buoyancy, background_fields,
            velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcings.v;
            exclude_periphery)

    launch!(arch, grid, kernel_parameters, compute_Gw!,
            tendencies.w, grid,
            advection, coriolis, stokes_drift, closure, w_bcs, buoyancy, background_fields,
            velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcings.w;
            exclude_periphery)

    foreach_name(tracers) do val_tracer_index, val_tracer_name
        launch_tracer_tendency!(model, kernel_parameters, val_tracer_index, val_tracer_name)
    end

    return nothing
end

@inline function launch_tracer_tendency!(model, kernel_parameters, ::Val{tracer_index}, ::Val{tracer_name}) where {tracer_index, tracer_name}
    arch = model.architecture
    grid = model.grid

    @inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
    @inbounds c_advection   = model.advection[tracer_name]
    @inbounds forcing       = model.forcing[tracer_name]
    @inbounds c_bcs = NamedTuple(model.tracers[tracer_name].boundary_conditions)

    launch!(arch, grid, kernel_parameters, compute_Gc!,
            c_tendency, grid,
            Val(tracer_index), Val(tracer_name), c_advection, model.closure, c_bcs, model.buoyancy,
            model.biogeochemistry, model.background_fields, model.velocities, model.tracers, model.auxiliary_fields,
            model.closure_fields, model.clock, forcing)

    return nothing
end

function compute_biogeochemical_transitions!(model, kernel_parameters)
    model_fields = merge(model.velocities, model.tracers, model.auxiliary_fields,
                         biogeochemical_auxiliary_fields(model.biogeochemistry))

    return add_biogeochemical_transitions!(model.timestepper.Gⁿ, model.biogeochemistry,
                                           model.grid, model.clock, model_fields; kernel_parameters)
end

#####
##### Tendency calculators for u, v, w-velocity
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function compute_Gu!(Gu, grid,
                             advection, coriolis, stokes_drift, closure, u_bcs, buoyancy, background_fields,
                             velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, grid,
                                                advection, coriolis, stokes_drift, closure, u_bcs, buoyancy, background_fields,
                                                velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function compute_Gv!(Gv, grid,
                             advection, coriolis, stokes_drift, closure, v_bcs, buoyancy, background_fields,
                             velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, grid,
                                                advection, coriolis, stokes_drift, closure, v_bcs, buoyancy, background_fields,
                                                velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
end

""" Calculate the right-hand-side of the w-velocity equation. """
@kernel function compute_Gw!(Gw, grid,
                             advection, coriolis, stokes_drift, closure, w_bcs, buoyancy, background_fields,
                             velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
    i, j, k = @index(Global, NTuple)
    @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, grid,
                                                advection, coriolis, stokes_drift, closure, w_bcs, buoyancy, background_fields,
                                                velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function compute_Gc!(Gc, grid,
                             val_index, val_tracer_name, advection, closure, c_bcs, buoyancy,
                             biogeochemistry, background_fields,
                             velocities, tracers, auxiliary_fields, closure_fields,
                             clock, forcing)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid,
                                            val_index, val_tracer_name, advection, closure, c_bcs, buoyancy,
                                            biogeochemistry, background_fields,
                                            velocities, tracers, auxiliary_fields, closure_fields,
                                            clock, forcing)
end
