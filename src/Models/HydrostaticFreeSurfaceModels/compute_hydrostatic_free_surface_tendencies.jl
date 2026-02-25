import Oceananigans: tracer_tendency_kernel_function
import Oceananigans.Models: interior_tendency_kernel_parameters

using Oceananigans: fields, prognostic_fields, TendencyCallsite, UpdateStateCallsite
using Oceananigans.Grids: halo_size
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Biogeochemistry: update_tendencies!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: FlavorOfCATKE, FlavorOfTD

using Oceananigans.Grids: get_active_cells_map
using Oceananigans.Advection: fixed_order_scheme
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

    grid = model.grid
    arch = architecture(grid)

    active_cells_map = get_active_cells_map(model.grid, Val(:interior))
		
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

    grid = model.grid
    arch = architecture(grid)

    active_cells_map  = get_active_cells_map(model.grid, Val(:interior))
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

        condition_maps = model.condition_maps[tracer_name]
        @inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection   = model.advection[tracer_name]
        @inbounds c_forcing     = model.forcing[tracer_name]
        @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])

        pre_args = (Val(tracer_index), Val(tracer_name))
        post_args = (model.closure,
                     c_immersed_bc,
                     model.buoyancy,
                     model.biogeochemistry,
                     model.transport_velocities,
                     model.free_surface,
                     model.tracers,
                     model.closure_fields,
                     model.auxiliary_fields,
                     model.clock,
                     c_forcing )

        args = prepend_args(pre_args, generate_kernel_args(c_advection, post_args; condition_maps))

        launch!(arch, grid, kernel_parameters,
                compute_hydrostatic_free_surface_Gc!,
                c_tendency,
                grid,
                args;
                active_cells_map=condition_maps)
    end

    return nothing
end


prepend_args(args1, args2) = (args1..., args2...)

function prepend_args(args1, args2::NamedTuple)
    new_args = Dict()
    
    for key in keys(args2)
        new_args[key] = (args1..., args2[key]...)
    end
    return (; new_args...)
end

""" Generate arguments for each mapped condition """
function generate_kernel_args(scheme, common_args; condition_maps=nothing)
    if isnothing(condition_maps)
        return (scheme, common_args...)
    else
        static_advection = fixed_order_scheme(scheme)
        return (; interior = (static_advection, common_args...),
                  boundary = (scheme, common_args...))
    end
end

"""
    compute_hydrostatic_momentum_tendencies!(model, velocities, kernel_parameters; active_cells_map=nothing)

Compute momentum tendencies for `u` and `v` in the grid interior (or on specified active cells).
"""
function compute_hydrostatic_momentum_tendencies!(model, velocities, kernel_parameters; active_cells_map=nothing)

    grid = model.grid
    arch = architecture(grid)

    momentum_condition_maps = model.condition_maps.momentum

    u_immersed_bc = immersed_boundary_condition(velocities.u)
    v_immersed_bc = immersed_boundary_condition(velocities.v)

    u_forcing = model.forcing.u
    v_forcing = model.forcing.v

    end_momentum_kernel_args = (velocities,
                                model.free_surface,
                                model.tracers,
                                model.buoyancy,
                                model.closure_fields,
                                model.pressure.pHY′,
                                model.auxiliary_fields,
                                model.vertical_coordinate,
                                model.clock)


    u_kernel_args = tuple(model.coriolis, model.closure, u_immersed_bc, end_momentum_kernel_args..., u_forcing)
    u_kernel_args_tuple = generate_kernel_args(model.advection.momentum, u_kernel_args; condition_maps=momentum_condition_maps)

    v_kernel_args = tuple(model.coriolis, model.closure, v_immersed_bc, end_momentum_kernel_args..., v_forcing)
    v_kernel_args_tuple = generate_kernel_args(model.advection.momentum, v_kernel_args; condition_maps=momentum_condition_maps)

    launch!(arch,
            grid,
            kernel_parameters,
            compute_hydrostatic_free_surface_Gu!,
            model.timestepper.Gⁿ.u,
            grid,
            u_kernel_args_tuple;
            active_cells_map=momentum_condition_maps)

    launch!(arch,
            grid, 
            kernel_parameters, 
            compute_hydrostatic_free_surface_Gv!, 
	    model.timestepper.Gⁿ.v,
	    grid,
            v_kernel_args_tuple; 
            active_cells_map=momentum_condition_maps)

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

