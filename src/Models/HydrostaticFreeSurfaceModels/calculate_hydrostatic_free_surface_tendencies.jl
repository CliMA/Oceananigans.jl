using Oceananigans: fields, prognostic_fields, TendencyCallsite, UpdateStateCallsite
using Oceananigans.Utils: work_layout
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Biogeochemistry: update_tendencies!

import Oceananigans.TimeSteppers: calculate_tendencies!
import Oceananigans: tracer_tendency_kernel_function

using Oceananigans.ImmersedBoundaries: use_only_active_cells, ActiveCellsIBG, active_linear_index_to_ntuple

"""
    calculate_tendencies!(model::HydrostaticFreeSurfaceModel, callbacks)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model::HydrostaticFreeSurfaceModel, callbacks)

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    calculate_hydrostatic_free_surface_interior_tendency_contributions!(model)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_hydrostatic_boundary_tendency_contributions!(model.timestepper.Gⁿ,
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
using Oceananigans.TurbulenceClosures.MEWSVerticalDiffusivities: MEWS

@inline tracer_tendency_kernel_function(model::HFSM, name, c, K)                     = calculate_hydrostatic_free_surface_Gc!, c, K
@inline tracer_tendency_kernel_function(model::HFSM, ::Val{:K}, c::MEWS,          K) = calculate_hydrostatic_free_surface_Ge!, c, K
@inline tracer_tendency_kernel_function(model::HFSM, ::Val{:e}, c::FlavorOfCATKE, K) = calculate_hydrostatic_free_surface_Ge!, c, K

function tracer_tendency_kernel_function(model::HFSM, ::Val{:e}, closures::Tuple, diffusivity_fields::Tuple)
    catke_index = findfirst(c -> c isa FlavorOfCATKE, closures)

    if isnothing(catke_index)
        return calculate_hydrostatic_free_surface_Gc!, closures, diffusivity_fields
    else
        catke_closure = closures[catke_index]
        catke_diffusivity_fields = diffusivity_fields[catke_index]
        return calculate_hydrostatic_free_surface_Ge!, catke_closure, catke_diffusivity_fields 
    end
end

function tracer_tendency_kernel_function(model::HFSM, ::Val{:K}, closures::Tuple, diffusivity_fields::Tuple)
    mews_index = findfirst(c -> c isa MEWS, closures)

    if isnothing(mews_index)
        return calculate_hydrostatic_free_surface_Gc!, closures, diffusivity_fields
    else
        mews_closure = closures[mews_index]
        mews_diffusivity_fields = diffusivity_fields[mews_index]
        return  calculate_hydrostatic_free_surface_Ge!, mews_closure, mews_diffusivity_fields 
    end
end

top_tracer_boundary_conditions(grid, tracers) =
    NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

""" Store previous value of the source term and calculate current source term. """
function calculate_hydrostatic_free_surface_interior_tendency_contributions!(model)

    arch = model.architecture
    grid = model.grid

    calculate_hydrostatic_momentum_tendencies!(model, model.velocities)

    top_tracer_bcs = top_tracer_boundary_conditions(grid, model.tracers)
    only_active_cells = use_only_active_cells(grid)

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))
        c_tendency    = model.timestepper.Gⁿ[tracer_name]
        c_advection   = model.advection[tracer_name]
        c_forcing     = model.forcing[tracer_name]
        c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])

        tendency_kernel!, closure, diffusivity = tracer_tendency_kernel_function(model, Val(tracer_name), model.closure, model.diffusivity_fields)

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

        launch!(arch, grid, :xyz,
                tendency_kernel!,
                c_tendency,
                grid,
                args;
                only_active_cells)
    end

    return nothing
end

#####
##### Boundary condributions to hydrostatic free surface model
#####

function apply_flux_bcs!(Gcⁿ, c, arch, args...)
    apply_x_bcs!(Gcⁿ, c, arch, args...)
    apply_y_bcs!(Gcⁿ, c, arch, args...)
    apply_z_bcs!(Gcⁿ, c, arch, args...)

    return nothing
end

function calculate_free_surface_tendency!(grid, model)

    arch = architecture(grid)

    args = tuple(model.velocities,
                 model.free_surface,
                 model.tracers,
                 model.auxiliary_fields,
                 model.forcing,
                 model.clock)

    launch!(arch, grid, :xy,
            calculate_hydrostatic_free_surface_Gη!, model.timestepper.Gⁿ.η,
            grid, args)

    return nothing
end
    

""" Calculate momentum tendencies if momentum is not prescribed."""
function calculate_hydrostatic_momentum_tendencies!(model, velocities)

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
    
    only_active_cells = use_only_active_cells(grid)

    launch!(arch, grid, :xyz,
            calculate_hydrostatic_free_surface_Gu!, model.timestepper.Gⁿ.u, grid, u_kernel_args;
            only_active_cells)

    launch!(arch, grid, :xyz,
            calculate_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, grid, v_kernel_args;
            only_active_cells)

    calculate_free_surface_tendency!(grid, model)

    return nothing
end

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_hydrostatic_boundary_tendency_contributions!(Gⁿ, arch, velocities, free_surface, tracers, args...)

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
@kernel function calculate_hydrostatic_free_surface_Gu!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

@kernel function calculate_hydrostatic_free_surface_Gu!(Gu, grid::ActiveCellsIBG, args)
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_ntuple(idx, grid)
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function calculate_hydrostatic_free_surface_Gv!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

@kernel function calculate_hydrostatic_free_surface_Gv!(Gv, grid::ActiveCellsIBG, args)
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_ntuple(idx, grid)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for tracers
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_hydrostatic_free_surface_Gc!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

@kernel function calculate_hydrostatic_free_surface_Gc!(Gc, grid::ActiveCellsIBG, args)
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_ntuple(idx, grid)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the subgrid scale energy equation. """
@kernel function calculate_hydrostatic_free_surface_Ge!(Ge, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Ge[i, j, k] = hydrostatic_turbulent_kinetic_energy_tendency(i, j, k, grid, args...)
end

@kernel function calculate_hydrostatic_free_surface_Ge!(Ge, grid::ActiveCellsIBG, args)
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_ntuple(idx, grid)
    @inbounds Ge[i, j, k] = hydrostatic_turbulent_kinetic_energy_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for an explicit free surface
#####

""" Calculate the right-hand-side of the free surface displacement (``η``) equation. """
@kernel function calculate_hydrostatic_free_surface_Gη!(Gη, grid, args)
    i, j = @index(Global, NTuple)
    @inbounds Gη[i, j, grid.Nz+1] = free_surface_tendency(i, j, grid, args...)
end

