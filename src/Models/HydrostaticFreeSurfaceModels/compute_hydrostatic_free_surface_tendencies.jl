import Oceananigans: tracer_tendency_kernel_function
import Oceananigans.TimeSteppers: compute_tendencies!
import Oceananigans.Models: complete_communication_and_compute_boundary!
import Oceananigans.Models: interior_tendency_kernel_parameters

using Oceananigans: fields, prognostic_fields, TendencyCallsite, UpdateStateCallsite
using Oceananigans.Utils: work_layout, KernelParameters
using Oceananigans.Grids: halo_size
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Biogeochemistry: update_tendencies!
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: FlavorOfCATKE

using Oceananigans.ImmersedBoundaries: active_interior_map, ActiveCellsIBG, 
                                       InteriorMap, active_linear_index_to_tuple

using KernelAbstractions: @private, @uniform, @groupsize, @index, @localmem


"""
    compute_tendencies!(model::HydrostaticFreeSurfaceModel, callbacks)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function compute_tendencies!(model::HydrostaticFreeSurfaceModel, callbacks)

    kernel_parameters = tuple(interior_tendency_kernel_parameters(model.grid))

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters;
                                                             active_cells_map = active_interior_map(model.grid))

    complete_communication_and_compute_boundary!(model, model.grid, model.architecture)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    compute_hydrostatic_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                                         model.architecture,
                                                         model.velocities,
                                                         model.free_surface,
                                                         model.tracers,
                                                         model.clock,
                                                         fields(model),
                                                         model.closure,
                                                         model.buoyancy)

    for callback in callbacks
        callback.callsite isa TendencyCallsite && callback(model)
    end

    update_tendencies!(model.biogeochemistry, model)

    return nothing
end

@inline function top_tracer_boundary_conditions(grid, tracers)
    names = propertynames(tracers)
    values = Tuple(tracers[c].boundary_conditions.top for c in names)

    # Some shenanigans for type stability?
    return NamedTuple{tuple(names...)}(tuple(values...))
end

""" Store previous value of the source term and compute current source term. """
function compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters; active_cells_map = nothing)

    arch = model.architecture
    grid = model.grid

    compute_hydrostatic_momentum_tendencies!(model, model.velocities, kernel_parameters; active_cells_map)

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))

        @inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection   = model.advection[tracer_name]
        @inbounds c_forcing     = model.forcing[tracer_name]
        @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])

        args = tuple(Val(tracer_index),
                     Val(tracer_name),
                     c_advection,
                     model.closure,
                     c_immersed_bc,
                     model.buoyancy,
                     model.biogeochemistry,
                     model.velocities,
                     model.free_surface,
                     model.tracers,
                     model.diffusivity_fields,
                     model.auxiliary_fields,
                     c_forcing,
                     model.clock)

        for parameters in kernel_parameters
            launch!(arch, grid, parameters,
                    compute_hydrostatic_free_surface_Gc!,
                    c_tendency,
                    grid,
                    active_cells_map,
                    args;
                    active_cells_map)
        end
    end

    return nothing
end

#####
##### Boundary condributions to hydrostatic free surface model
#####

function apply_flux_bcs!(Gcⁿ, c, arch, args)
    apply_x_bcs!(Gcⁿ, c, arch, args...)
    apply_y_bcs!(Gcⁿ, c, arch, args...)
    apply_z_bcs!(Gcⁿ, c, arch, args...)
    return nothing
end

function compute_free_surface_tendency!(grid, model, kernel_parameters)

    arch = architecture(grid)

    args = tuple(model.velocities,
                 model.free_surface,
                 model.tracers,
                 model.auxiliary_fields,
                 model.forcing,
                 model.clock)

    launch!(arch, grid, kernel_parameters,
            compute_hydrostatic_free_surface_Gη!, model.timestepper.Gⁿ.η, 
            grid, args)

    return nothing
end

""" Calculate momentum tendencies if momentum is not prescribed."""
function compute_hydrostatic_momentum_tendencies!(model, velocities, kernel_parameters; active_cells_map = nothing)

    grid = model.grid
    arch = architecture(grid)

    u_immersed_bc = immersed_boundary_condition(velocities.u)
    v_immersed_bc = immersed_boundary_condition(velocities.v)

    start_momentum_kernel_args = (model.advection.momentum,
                                  model.coriolis,
                                  model.closure)

    end_momentum_kernel_args = (velocities,
                                model.free_surface,
                                model.tracers,
                                model.buoyancy,
                                model.diffusivity_fields,
                                model.pressure.pHY′,
                                model.auxiliary_fields,
                                model.forcing,
                                model.clock)

    u_kernel_args = tuple(start_momentum_kernel_args..., u_immersed_bc, end_momentum_kernel_args...)
    v_kernel_args = tuple(start_momentum_kernel_args..., v_immersed_bc, end_momentum_kernel_args...)

    for parameters in kernel_parameters
        launch!(arch, grid, parameters,
                compute_hydrostatic_free_surface_Gu!, model.timestepper.Gⁿ.u, grid, 
                active_cells_map, u_kernel_args;
                active_cells_map)

        launch!(arch, grid, parameters,
                compute_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, grid, 
                active_cells_map, v_kernel_args;
                active_cells_map)
    end

    compute_free_surface_tendency!(grid, model, :xy)

    return nothing
end

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function compute_hydrostatic_boundary_tendency_contributions!(Gⁿ, arch, velocities, free_surface, tracers, args...)

    args = Tuple(args)

    # Velocity fields
    for i in (:u, :v)
        apply_flux_bcs!(Gⁿ[i], velocities[i], arch, args)
    end

    # Free surface
    apply_flux_bcs!(Gⁿ.η, displacement(free_surface), arch, args)

    # Tracer fields
    for i in propertynames(tracers)
        apply_flux_bcs!(Gⁿ[i], tracers[i], arch, args)
    end

    return nothing
end

#####
##### Tendency calculators for u, v
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function compute_hydrostatic_free_surface_Gu!(Gu, grid, map, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

@kernel function compute_hydrostatic_free_surface_Gu!(Gu, grid::ActiveCellsIBG, map, args)
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_tuple(idx, map, grid)
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function compute_hydrostatic_free_surface_Gv!(Gv, grid, map, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

@kernel function compute_hydrostatic_free_surface_Gv!(Gv, grid::ActiveCellsIBG, map, args)
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_tuple(idx, map, grid)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for tracers
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function compute_hydrostatic_free_surface_Gc!(Gc, grid, map, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

@kernel function compute_hydrostatic_free_surface_Gc!(Gc, grid::ActiveCellsIBG, map, args)
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_tuple(idx, map, grid)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for an explicit free surface
#####

""" Calculate the right-hand-side of the free surface displacement (``η``) equation. """
@kernel function compute_hydrostatic_free_surface_Gη!(Gη, grid, args)
    i, j = @index(Global, NTuple)
    @inbounds Gη[i, j, grid.Nz+1] = free_surface_tendency(i, j, grid, args...)
end
