import Oceananigans: tracer_tendency_kernel_function
import Oceananigans.Models: interior_tendency_kernel_parameters, update_model_field_time_series!,
                           surface_kernel_parameters, volume_kernel_parameters
import Oceananigans.TimeSteppers: compute_tendencies!

using Oceananigans: fields, prognostic_fields, TendencyCallsite, UpdateStateCallsite
using Oceananigans.BoundaryConditions: fill_halo_regions!, update_boundary_conditions!
using Oceananigans.BuoyancyFormulations: compute_buoyancy_gradients!
using Oceananigans.Grids: halo_size
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Biogeochemistry: update_tendencies!, biogeochemical_drift_velocity
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!
using Oceananigans.TurbulenceClosures: closure_auxiliary_velocity, compute_closure_fields!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: FlavorOfCATKE, FlavorOfTD

using Oceananigans.Utils: get_active_cells_map

function compute_tendencies!(model::HydrostaticFreeSurfaceModel, callbacks)
    compute_momentum_tendencies!(model, callbacks)
    compute_tracer_tendencies!(model)
    return nothing
end

function refresh_hydrostatic_tendency_diagnostic_state!(model::HydrostaticFreeSurfaceModel)
    grid = model.grid
    arch = architecture(grid)

    @apply_regionally begin
        compute_auxiliary_fields!(model.auxiliary_fields)
        update_boundary_conditions!(fields(model), model)
        update_prescribed_velocity_field_operations!(model.velocities)
    end

    fill_halo_regions!((model.velocities.u, model.velocities.v), model.clock, fields(model); async=false)
    fill_halo_regions!(model.tracers, model.clock, fields(model); async=false)
    compute_auxiliary_fields!(model.auxiliary_fields)

    @apply_regionally begin
        surface_params = surface_kernel_parameters(grid)
        volume_params = volume_kernel_parameters(grid)
        κ_params = diffusivity_kernel_parameters(grid)
        compute_buoyancy_gradients!(model.buoyancy, grid, model.tracers, parameters=volume_params)
        update_vertical_velocities!(model.velocities, grid, model, parameters=surface_params)
        update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers, parameters=surface_params)
        compute_closure_fields!(model.closure_fields, model.closure, model, parameters=κ_params)
    end

    fill_hydrostatic_closure_field_halos!(model.closure_fields, grid)
    fill_hydrostatic_pressure_halos!(model.pressure.pHY′, grid)
    compute_auxiliary_fields!(model.auxiliary_fields)
    fill_halo_regions!(model.velocities, model.clock, fields(model); async=false)

    return nothing
end

"""
    compute_momentum_tendencies!(model::HydrostaticFreeSurfaceModel, callbacks)

Compute tendencies for horizontal velocity fields `u` and `v`.

This function:
1. Computes interior momentum tendencies (advection, Coriolis, pressure gradient, diffusion, forcing)
2. Completes halo communication and computes buffer tendencies for distributed grids
3. Computes flux boundary condition contributions
4. Executes any callbacks with `TendencyCallsite`

Momentum tendencies are stored in `model.timestepper.Gⁿ.u` and `model.timestepper.Gⁿ.v`.
"""
function compute_momentum_tendencies!(model::HydrostaticFreeSurfaceModel, callbacks)
    update_model_field_time_series!(model, model.clock)
    refresh_momentum_advection_state!(model, model.velocities)
    refresh_hydrostatic_tendency_diagnostic_state!(model)

    grid = model.grid
    arch = architecture(grid)

    active_cells_map = get_active_cells_map(model.grid, Val(:core))
    kernel_parameters = interior_tendency_kernel_parameters(arch, grid)

    compute_hydrostatic_momentum_tendencies!(model, model.velocities, kernel_parameters; active_cells_map)
    complete_communication_and_compute_momentum_buffer!(model, grid, arch)

    for callback in callbacks
        callback.callsite isa TendencyCallsite && callback(model)
    end

    return nothing
end

"""
    compute_tracer_tendencies!(model::HydrostaticFreeSurfaceModel)

Compute tendencies for all tracer fields.

This function:
1. Computes interior tracer tendencies (advection, diffusion, forcing, biogeochemistry sources)
2. Completes halo communication and computes buffer tendencies for distributed grids
3. Computes flux boundary condition contributions
4. Scales tendencies by the grid stretching factor for z-star coordinates
5. Updates biogeochemistry tendencies

Tracers are advected using `model.transport_velocities` which may differ from `model.velocities`
when using split-explicit free surfaces (transport velocities include barotropic correction).

Tracer tendencies are stored in `model.timestepper.Gⁿ[tracer_name]`.
"""
function compute_tracer_tendencies!(model::HydrostaticFreeSurfaceModel)
    update_model_field_time_series!(model, model.clock)
    refresh_transport_advection_state!(model, model.velocities)
    refresh_hydrostatic_tendency_diagnostic_state!(model)
    refresh_update_state_tracer_advection_halos!(model)

    grid = model.grid
    arch = architecture(grid)

    active_cells_map  = get_active_cells_map(model.grid, Val(:core))
    kernel_parameters = interior_tendency_kernel_parameters(arch, grid)

    compute_hydrostatic_tracer_tendencies!(model, kernel_parameters; active_cells_map)
    complete_communication_and_compute_tracer_buffer!(model, grid, arch)
    compute_tracer_flux_bcs!(model)

    scale_by_stretching_factor!(model.timestepper.Gⁿ, model.tracers, model.grid)

    update_tendencies!(model.biogeochemistry, model)

    return nothing
end

# Fallback
compute_free_surface_tendency!(grid, model, free_surface) = nothing

@inline function top_tracer_boundary_conditions(grid, tracers)
    names = propertynames(tracers)
    values = Tuple(tracers[c].boundary_conditions.top for c in names)

    # Some shenanigans for type stability?
    return NamedTuple{tuple(names...)}(tuple(values...))
end

"""
    compute_hydrostatic_tracer_tendencies!(model, kernel_parameters; active_cells_map=nothing)

Compute tracer tendencies in the grid interior (or on specified active cells).

Launches the tracer tendency kernel for each tracer, computing advection, diffusion,
and forcing contributions. Uses `model.transport_velocities` for advection.
"""
function compute_hydrostatic_tracer_tendencies!(model, kernel_parameters; active_cells_map=nothing)

    arch = model.architecture
    grid = model.grid

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))

        @inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection   = model.advection[tracer_name]
        @inbounds c_forcing     = model.forcing[tracer_name]
        @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])
        tracer_name_val = Val(tracer_name)

        biogeochemical_velocities = biogeochemical_drift_velocity(model.biogeochemistry, tracer_name_val)
        closure_velocities = closure_auxiliary_velocity(model.closure, model.closure_fields, tracer_name_val)

        refresh_tracer_auxiliary_velocity_halos!(biogeochemical_velocities)
        refresh_tracer_auxiliary_velocity_halos!(closure_velocities)
        refresh_tracer_advective_forcing_halos!(c_forcing)

        auxiliary_velocities = tracer_auxiliary_velocities(biogeochemical_velocities,
                                                           closure_velocities,
                                                           c_forcing)

        args = tuple(Val(tracer_index),
                     tracer_name_val,
                     c_advection,
                     model.closure,
                     c_immersed_bc,
                     model.buoyancy,
                     model.biogeochemistry,
                     model.transport_velocities,
                     auxiliary_velocities,
                     model.velocities,
                     model.free_surface,
                     model.tracers,
                     model.closure_fields,
                     model.auxiliary_fields,
                     model.clock,
                     c_forcing)

        launch!(arch, grid, kernel_parameters,
                compute_hydrostatic_free_surface_Gc!,
                c_tendency,
                grid,
                args;
                active_cells_map)
    end

    return nothing
end

"""
    compute_hydrostatic_momentum_tendencies!(model, velocities, kernel_parameters; active_cells_map=nothing)

Compute momentum tendencies for `u` and `v` in the grid interior (or on specified active cells).
"""
function compute_hydrostatic_momentum_tendencies!(model, velocities, kernel_parameters; active_cells_map=nothing)

    grid = model.grid
    arch = architecture(grid)

    u_immersed_bc = immersed_boundary_condition(velocities.u)
    v_immersed_bc = immersed_boundary_condition(velocities.v)

    u_forcing = model.forcing.u
    v_forcing = model.forcing.v

    refresh_tracer_advective_forcing_halos!(u_forcing)
    refresh_tracer_advective_forcing_halos!(v_forcing)

    start_momentum_kernel_args = (model.advection.momentum,
                                  model.coriolis,
                                  model.closure)

    end_momentum_kernel_args = (velocities,
                                model.transport_velocities,
                                model.free_surface,
                                model.tracers,
                                model.buoyancy,
                                model.closure_fields,
                                model.pressure.pHY′,
                                model.auxiliary_fields,
                                model.vertical_coordinate,
                                model.clock)

    u_kernel_args = tuple(start_momentum_kernel_args..., u_immersed_bc, end_momentum_kernel_args..., u_forcing)
    v_kernel_args = tuple(start_momentum_kernel_args..., v_immersed_bc, end_momentum_kernel_args..., v_forcing)

    launch!(arch, grid, kernel_parameters,
            compute_hydrostatic_free_surface_Gu!, model.timestepper.Gⁿ.u, grid,
            u_kernel_args; active_cells_map)

    launch!(arch, grid, kernel_parameters,
            compute_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, grid,
            v_kernel_args; active_cells_map)

    return nothing
end

#####
##### Tendency calculators for u, v
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function compute_hydrostatic_free_surface_Gu!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function compute_hydrostatic_free_surface_Gv!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for tracers
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function compute_hydrostatic_free_surface_Gc!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end
