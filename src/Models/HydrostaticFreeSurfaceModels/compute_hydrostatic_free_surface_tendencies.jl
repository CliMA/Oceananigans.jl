import Oceananigans.TimeSteppers: compute_tendencies!
import Oceananigans: tracer_tendency_kernel_function

using Oceananigans.Utils: work_layout, KernelParameters
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Grids: halo_size
using Oceananigans: fields, prognostic_fields, TendencyCallsite, UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_tendencies!

import Oceananigans.TimeSteppers: compute_tendencies!
import Oceananigans: tracer_tendency_kernel_function

import Oceananigans.Models: complete_communication_and_compute_boundary!
import Oceananigans.Models: interior_tendency_kernel_parameters

using Oceananigans.ImmersedBoundaries: use_only_active_interior_cells, ActiveCellsIBG, 
                                       InteriorMap, active_linear_index_to_interior_tuple

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
                                                             only_active_cells = use_only_active_interior_cells(model.grid))

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

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: FlavorOfCATKE

@inline tracer_tendency_kernel_function(model::HFSM, name, c, K)                     = compute_hydrostatic_free_surface_Gc!, c, K
@inline tracer_tendency_kernel_function(model::HFSM, ::Val{:e}, c::FlavorOfCATKE, K) = compute_hydrostatic_free_surface_Ge!, c, K

function tracer_tendency_kernel_function(model::HFSM, ::Val{:e}, closures::Tuple, diffusivity_fields::Tuple)
    catke_index = findfirst(c -> c isa FlavorOfCATKE, closures)

    if isnothing(catke_index)
        return compute_hydrostatic_free_surface_Gc!, closures, diffusivity_fields
    else
        catke_closure = closures[catke_index]
        catke_diffusivity_fields = diffusivity_fields[catke_index]
        return compute_hydrostatic_free_surface_Ge!, catke_closure, catke_diffusivity_fields 
    end
end

@inline function top_tracer_boundary_conditions(grid, tracers)
    names = propertynames(tracers)
    values = Tuple(tracers[c].boundary_conditions.top for c in names)

    # Some shenanigans for type stability?
    return NamedTuple{tuple(names...)}(tuple(values...))
end

""" Store previous value of the source term and compute current source term. """
function compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters; only_active_cells = nothing)

    arch = model.architecture
    grid = model.grid

    compute_hydrostatic_momentum_tendencies!(model, model.velocities, kernel_parameters; only_active_cells)

    top_tracer_bcs = top_tracer_boundary_conditions(grid, model.tracers)

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))
        @inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection   = model.advection[tracer_name]
        @inbounds c_forcing     = model.forcing[tracer_name]
        @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])

        tendency_kernel!, closure, diffusivity = tracer_tendency_kernel_function(model,
                                                                                 Val(tracer_name),
                                                                                 model.closure,
                                                                                 model.diffusivity_fields)

        args = tuple(Val(tracer_index),
                     Val(tracer_name),
                     c_advection,
                     closure,
                     c_immersed_bc,
                     model.buoyancy,
                     model.biogeochemistry,
                     model.velocities,
                     model.free_surface,
                     model.tracers,
                     top_tracer_bcs,
                     diffusivity,
                     model.auxiliary_fields,
                     c_forcing,
                     model.clock)

        for parameters in kernel_parameters
            launch!(arch, grid, parameters,
                    tendency_kernel!,
                    c_tendency,
                    grid,
                    only_active_cells,
                    args...;
                    only_active_cells)
        end
    end

    return nothing
end

#####
##### Boundary condributions to hydrostatic free surface model
#####

function apply_flux_bcs!(Gcⁿ, c, arch, args::Vararg{Any, N}) where {N}
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
            grid, args...)

    return nothing
end

""" Calculate momentum tendencies if momentum is not prescribed."""
function compute_hydrostatic_momentum_tendencies!(model, velocities, kernel_parameters; only_active_cells = nothing)

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
                only_active_cells, u_kernel_args...;
                only_active_cells)

        launch!(arch, grid, parameters,
                compute_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, grid, 
                only_active_cells, v_kernel_args...;
                only_active_cells)
    end

    compute_free_surface_tendency!(grid, model, :xy)

    return nothing
end

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function compute_hydrostatic_boundary_tendency_contributions!(Gⁿ, arch, velocities, free_surface, tracers, args...)

    args = Tuple(args)

    # Velocity fields
    for i in (:u, :v)
        apply_flux_bcs!(Gⁿ[i], velocities[i], arch, args...)
    end

    # Free surface
    apply_flux_bcs!(Gⁿ.η, displacement(free_surface), arch, args...)

    # Tracer fields
    for i in propertynames(tracers)
        apply_flux_bcs!(Gⁿ[i], tracers[i], arch, args...)
    end

    return nothing
end

#####
##### Tendency calculators for u, v
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function compute_hydrostatic_free_surface_Gu!(Gu, grid, interior_map, args::Vararg{Any, N}) where {N}
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

@kernel function compute_hydrostatic_free_surface_Gu!(Gu, grid::ActiveCellsIBG, ::InteriorMap, args::Vararg{Any, N}) where {N}
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_interior_tuple(idx, grid)
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function compute_hydrostatic_free_surface_Gv!(Gv, grid, interior_map, args::Vararg{Any, N}) where {N}
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

@kernel function compute_hydrostatic_free_surface_Gv!(Gv, grid::ActiveCellsIBG, ::InteriorMap, args::Vararg{Any, N}) where {N}
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_interior_tuple(idx, grid)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for tracers
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function compute_hydrostatic_free_surface_Gc!(Gc, grid, interior_map, args::Vararg{Any, N}) where {N}
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

@kernel function compute_hydrostatic_free_surface_Gc!(Gc, grid::ActiveCellsIBG, ::InteriorMap, args::Vararg{Any, N}) where {N}
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_interior_tuple(idx, grid)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the subgrid scale energy equation. """
@kernel function compute_hydrostatic_free_surface_Ge!(Ge, grid, interior_map, args::Vararg{Any, N}) where {N}
    i, j, k = @index(Global, NTuple)
    @inbounds Ge[i, j, k] = hydrostatic_turbulent_kinetic_energy_tendency(i, j, k, grid, args...)
end

@kernel function compute_hydrostatic_free_surface_Ge!(Ge, grid::ActiveCellsIBG, ::InteriorMap, args::Vararg{Any, N}) where {N}
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_interior_tuple(idx, grid)
    @inbounds Ge[i, j, k] = hydrostatic_turbulent_kinetic_energy_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for an explicit free surface
#####

""" Calculate the right-hand-side of the free surface displacement (``η``) equation. """
@kernel function compute_hydrostatic_free_surface_Gη!(Gη, grid, args::Vararg{Any, N}) where {N}
    i, j = @index(Global, NTuple)
    @inbounds Gη[i, j, grid.Nz+1] = free_surface_tendency(i, j, grid, args...)
end
