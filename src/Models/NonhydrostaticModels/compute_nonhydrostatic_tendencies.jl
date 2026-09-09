using Oceananigans: fields, prognostic_fields, TendencyCallsite
using Oceananigans.Biogeochemistry: update_tendencies!
using Oceananigans.Models: complete_communication_and_compute_buffer!, interior_tendency_kernel_parameters
using Oceananigans.Utils: get_active_cells_map

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
    active_cells_map  = get_active_cells_map(model.grid, Val(:core))

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
    u_immersed_bc        = velocities.u.boundary_conditions.immersed
    v_immersed_bc        = velocities.v.boundary_conditions.immersed
    w_immersed_bc        = velocities.w.boundary_conditions.immersed

    exclude_periphery = true
    launch!(arch, grid, kernel_parameters, compute_Gu!,
            tendencies.u, grid,
            advection, coriolis, stokes_drift, closure, u_immersed_bc, buoyancy, background_fields,
            velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcings.u;
            active_cells_map, exclude_periphery)

    launch!(arch, grid, kernel_parameters, compute_Gv!,
            tendencies.v, grid,
            advection, coriolis, stokes_drift, closure, v_immersed_bc, buoyancy, background_fields,
            velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcings.v;
            active_cells_map, exclude_periphery)

    launch!(arch, grid, kernel_parameters, compute_Gw!,
            tendencies.w, grid,
            advection, coriolis, stokes_drift, closure, w_immersed_bc, buoyancy, background_fields,
            velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcings.w;
            active_cells_map, exclude_periphery)

    launch_tracer_tendencies!(model, kernel_parameters, active_cells_map, Val(1), Val(propertynames(tracers)))

    return nothing
end

@inline launch_tracer_tendencies!(model, kernel_parameters, active_cells_map, ::Val, ::Val{()}) = nothing

@inline function launch_tracer_tendencies!(model, kernel_parameters, active_cells_map, ::Val{tracer_index}, ::Val{tracer_names}) where {tracer_index, tracer_names}
    tracer_name = first(tracer_names)
    arch = model.architecture
    grid = model.grid

    @inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
    @inbounds c_advection   = model.advection[tracer_name]
    @inbounds forcing       = model.forcing[tracer_name]
    @inbounds c_immersed_bc = model.tracers[tracer_name].boundary_conditions.immersed

    launch!(arch, grid, kernel_parameters, compute_Gc!,
            c_tendency, grid,
            Val(tracer_index), Val(tracer_name), c_advection, model.closure, c_immersed_bc, model.buoyancy,
            model.biogeochemistry, model.background_fields, model.velocities, model.tracers, model.auxiliary_fields,
            model.closure_fields, model.clock, forcing;
            active_cells_map)

    launch_tracer_tendencies!(model, kernel_parameters, active_cells_map, Val(tracer_index + 1), Val(Base.tail(tracer_names)))

    return nothing
end

#####
##### Tendency calculators for u, v, w-velocity
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function compute_Gu!(Gu, grid,
                             advection, coriolis, stokes_drift, closure, u_immersed_bc, buoyancy, background_fields,
                             velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, grid,
                                                advection, coriolis, stokes_drift, closure, u_immersed_bc, buoyancy, background_fields,
                                                velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function compute_Gv!(Gv, grid,
                             advection, coriolis, stokes_drift, closure, v_immersed_bc, buoyancy, background_fields,
                             velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, grid,
                                                advection, coriolis, stokes_drift, closure, v_immersed_bc, buoyancy, background_fields,
                                                velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
end

""" Calculate the right-hand-side of the w-velocity equation. """
@kernel function compute_Gw!(Gw, grid,
                             advection, coriolis, stokes_drift, closure, w_immersed_bc, buoyancy, background_fields,
                             velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
    i, j, k = @index(Global, NTuple)
    @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, grid,
                                                advection, coriolis, stokes_drift, closure, w_immersed_bc, buoyancy, background_fields,
                                                velocities, tracers, auxiliary_fields, closure_fields, hydrostatic_pressure, clock, forcing)
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function compute_Gc!(Gc, grid,
                             val_index, val_tracer_name, advection, closure, c_immersed_bc, buoyancy,
                             biogeochemistry, background_fields, velocities, tracers, auxiliary_fields, closure_fields,
                             clock, forcing)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid,
                                            val_index, val_tracer_name, advection, closure, c_immersed_bc, buoyancy,
                                            biogeochemistry, background_fields, velocities, tracers, auxiliary_fields, closure_fields,
                                            clock, forcing)
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

"""
$(TYPEDSIGNATURES)

Apply boundary conditions by adding flux divergences to the right-hand-side.
"""
function Oceananigans.TimeSteppers.compute_flux_bc_tendencies!(model::NonhydrostaticModel)
    names = Val(keys(prognostic_fields(model)))
    compute_flux_bcs!(compute_x_bcs!, model, names)
    compute_flux_bcs!(compute_y_bcs!, model, names)
    compute_flux_bcs!(compute_z_bcs!, model, names)
    return nothing
end

@inline compute_flux_bcs!(compute_bcs!, model, ::Val{()}) = nothing

# `fields(model)` is rebuilt at every level: passing the merged tuple down the recursion allocates
@inline function compute_flux_bcs!(compute_bcs!, model, ::Val{names}) where names
    name = first(names)
    Gc = model.timestepper.Gⁿ[name]
    c = prognostic_fields(model)[name]
    compute_bcs!(Gc, c, model.architecture, model.clock, fields(model))
    compute_flux_bcs!(compute_bcs!, model, Val(Base.tail(names)))
    return nothing
end
