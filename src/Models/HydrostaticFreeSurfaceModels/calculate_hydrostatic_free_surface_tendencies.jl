import Oceananigans.TimeSteppers: calculate_tendencies!
import Oceananigans: tracer_tendency_kernel_function

using Oceananigans.Architectures: device_event
using Oceananigans: fields, prognostic_fields
using Oceananigans.Utils: work_layout
using Oceananigans.Fields: immersed_boundary_condition

"""
    calculate_tendencies!(model::NonhydrostaticModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model::HydrostaticFreeSurfaceModel, fill_halo_events)

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    interior_events = calculate_hydrostatic_tendency_contributions!(model, :interior, barrier = device_event(arch))
    
    boundary_events = []
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :west,   barrier = fill_halo_events[end]))
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :east,   barrier = fill_halo_events[end]))
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :south,  barrier = fill_halo_events[end]))
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :north,  barrier = fill_halo_events[end]))
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :bottom, barrier = fill_halo_events[end]))
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :top,    barrier = fill_halo_events[end]))

    wait(device(model.architecture), MultiEvent(tuple(fill_halo_events..., interior_events..., boundary_events...)))

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_hydrostatic_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                                           model.grid,
                                                           model.architecture,
                                                           model.velocities,
                                                           model.free_surface,
                                                           model.tracers,
                                                           model.clock,
                                                           fields(model),
                                                           model.closure,
                                                           model.buoyancy)

    return nothing
end

function calculate_free_surface_tendency!(grid, model, dependencies)

    arch = architecture(grid)

    Gη_event = launch!(arch, grid, :xy,
                       calculate_hydrostatic_free_surface_Gη!, model.timestepper.Gⁿ.η,
                       grid,
                       model.velocities,
                       model.free_surface,
                       model.tracers,
                       model.auxiliary_fields,
                       model.forcing,
                       model.clock;
                       dependencies = dependencies)

    return Gη_event
end
    
@inline kernelsize(N, H, ::Val{:Interior}) = N .- H
@inline kernelsize(N, H, ::Val{:west})   = (H[1], N[2], N[3])
@inline kernelsize(N, H, ::Val{:east})   = (H[1], N[2], N[3])
@inline kernelsize(N, H, ::Val{:south})  = (N[1], H[2], N[3])
@inline kernelsize(N, H, ::Val{:north})  = (N[1], H[2], N[3])
@inline kernelsize(N, H, ::Val{:bottom}) = (N[1], N[2], H[3])
@inline kernelsize(N, H, ::Val{:top})    = (N[1], N[2], H[3])

@inline kerneloffsets(N, H, ::Val{:Interior}) = H
@inline kerneloffsets(N, H, ::Val{:west})   = (0, 0, 0)
@inline kerneloffsets(N, H, ::Val{:east})   = (N[1] - H[1], 0, 0)
@inline kerneloffsets(N, H, ::Val{:south})  = (0, 0, 0)
@inline kerneloffsets(N, H, ::Val{:north})  = (0, N[2] - H[2], 0)
@inline kerneloffsets(N, H, ::Val{:bottom}) = (0, 0, 0)
@inline kerneloffsets(N, H, ::Val{:top})    = (0, 0, N[3] - H[3])

""" Calculate momentum tendencies if momentum is not prescribed. `velocities` argument eases dispatch on `PrescribedVelocityFields`."""
function calculate_hydrostatic_momentum_tendencies!(model, velocities, kernel_size, offsets; dependencies = device_event(model))

    grid = model.grid
    arch = architecture(grid)

    u_immersed_bc = immersed_boundary_condition(velocities.u)
    v_immersed_bc = immersed_boundary_condition(velocities.v)

    start_momentum_kernel_args = (grid,
                                  model.advection.momentum,
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

    Gu_event = launch!(arch, grid, kernel_size,
                       calculate_hydrostatic_free_surface_Gu!, model.timestepper.Gⁿ.u, offsets, u_kernel_args...;
                       dependencies = dependencies)

    Gv_event = launch!(arch, grid, kernel_size,
                       calculate_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, offsets, v_kernel_args...;
                       dependencies = dependencies)

    Gη_event = calculate_free_surface_tendency!(grid, model, dependencies)

    events = [Gu_event, Gv_event, Gη_event]

    return events
end

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVD, CATKEVDArray

# Fallback
@inline tracer_tendency_kernel_function(model::HydrostaticFreeSurfaceModel, closure, tracer_name) =
    hydrostatic_free_surface_tracer_tendency

@inline tracer_tendency_kernel_function(model::HydrostaticFreeSurfaceModel, closure::CATKEVD, ::Val{:e}) =
    hydrostatic_turbulent_kinetic_energy_tendency

function tracer_tendency_kernel_function(model::HydrostaticFreeSurfaceModel, closures::Tuple, ::Val{:e})
    if any(cl isa Union{CATKEVD, CATKEVDArray} for cl in closures)
        return hydrostatic_turbulent_kinetic_energy_tendency
    else
        return hydrostatic_free_surface_tracer_tendency
    end
end

top_tracer_boundary_conditions(grid, tracers) =
    NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

""" Store previous value of the source term and calculate current source term. """
function calculate_hydrostatic_tendency_contributions!(model, region_to_compute; dependencies = device_event(model))

    arch = model.architecture
    grid = model.grid

    N = size(grid)
    H = halo_size(grid)
    kernel_size = kernelsize(N, H, Val(region_to_compute))
    offsets     = kerneloffsets(N, H, Val(region_to_compute))

    events = calculate_hydrostatic_momentum_tendencies!(model, model.velocities, kernel_size, offsets; dependencies)

    top_tracer_bcs = top_tracer_boundary_conditions(grid, model.tracers)

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))
        @inbounds c_tendency = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection = model.advection[tracer_name]
        @inbounds c_forcing = model.forcing[tracer_name]
        @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])
        c_kernel_function = tracer_tendency_kernel_function(model, model.closure, Val(tracer_name))

        Gc_event = launch!(arch, grid, kernel_size,
                           calculate_hydrostatic_free_surface_Gc!,
                           c_tendency,
                           c_kernel_function,
                           offsets,
                           grid,
                           Val(tracer_index),
                           c_advection,
                           model.closure,
                           c_immersed_bc,
                           model.buoyancy,
                           model.velocities,
                           model.free_surface,
                           model.tracers,
                           top_tracer_bcs,
                           model.diffusivity_fields,
                           model.auxiliary_fields,
                           c_forcing,
                           model.clock;
                           dependencies)

        push!(events, Gc_event)
    end

    return Tuple(events)
end

#####
##### Tendency calculators for u, v
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function calculate_hydrostatic_free_surface_Gu!(Gu, offsets, grid, args...)
    i, j, k = @index(Global, NTuple)
    i, j, k .+= offsets
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function calculate_hydrostatic_free_surface_Gv!(Gv, offsets, grid, args...)
    i, j, k = @index(Global, NTuple)
    i, j, k .+= offsets
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for tracers
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_hydrostatic_free_surface_Gc!(Gc, tendency_kernel_function, offsets, grid, args...)
    i, j, k = @index(Global, NTuple)
    i, j, k .+= offsets
    @inbounds Gc[i, j, k] = tendency_kernel_function(i, j, k, grid, args...)
end

#####
##### Tendency calculators for an explicit free surface
#####

""" Calculate the right-hand-side of the free surface displacement (η) equation. """
@kernel function calculate_hydrostatic_free_surface_Gη!(Gη, grid, args...)
    i, j = @index(Global, NTuple)
    @inbounds Gη[i, j, 1] = free_surface_tendency(i, j, grid, args...)
end

#####
##### Boundary condributions to hydrostatic free surface model
#####

function apply_flux_bcs!(Gcⁿ, events, c, arch, barrier, args...)
    x_bcs_event = apply_x_bcs!(Gcⁿ, c, arch, barrier, args...)
    y_bcs_event = apply_y_bcs!(Gcⁿ, c, arch, barrier, args...)
    z_bcs_event = apply_z_bcs!(Gcⁿ, c, arch, barrier, args...)

    push!(events, x_bcs_event, y_bcs_event, z_bcs_event)

    return nothing
end

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_hydrostatic_boundary_tendency_contributions!(Gⁿ, grid, arch, velocities, free_surface, tracers, args...)

    barrier = device_event(arch)

    events = []

    # Velocity fields
    for i in (:u, :v)
        apply_flux_bcs!(Gⁿ[i], events, velocities[i], arch, barrier, args...)
    end

    # Free surface
    apply_flux_bcs!(Gⁿ.η, events, displacement(free_surface), arch, barrier, args...)

    # Tracer fields
    for i in propertynames(tracers)
        apply_flux_bcs!(Gⁿ[i], events, tracers[i], arch, barrier, args...)
    end

    events = filter(e -> typeof(e) <: Event, events)

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end
