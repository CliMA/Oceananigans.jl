using Oceananigans: fields
using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Grids: active_interior_map
using Oceananigans.BoundaryConditions: apply_x_bcs!, apply_y_bcs!, apply_z_bcs!
using Oceananigans.TimeSteppers: store_field_tendencies!, ab2_step_field!, implicit_step!
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, immersed_∇_dot_qᶜ, hydrostatic_turbulent_kinetic_energy_tendency

function apply_flux_bcs!(Gcⁿ, c, arch, args)
    apply_x_bcs!(Gcⁿ, c, arch, args...)
    apply_y_bcs!(Gcⁿ, c, arch, args...)
    apply_z_bcs!(Gcⁿ, c, arch, args...)
    return nothing
end

function time_step_turbulent_kinetic_energy!(model)

    # 1. Compute new tendency.
    Gⁿe = model.timestepper.Gⁿ.e
    e = model.tracers.e
    arch = model.architecture
    grid = model.grid
    closure = model.closure
    previous_velocities = model.diffusivity_fields.previous_velocities

    args = tuple(closure,
                 model.velocities,
                 previous_velocities,
                 model.tracers,
                 model.buoyancy,
                 model.diffusivity_fields)

    launch!(arch, grid, :xyz,
            add_tke_source_terms!,
            Gⁿe, grid, args)

    # 2. Step forward
    Δt = model.clock.last_Δt
    χ = model.timestepper.χ
    G⁻e = model.timestepper.G⁻.e
    tracer_index = findfirst(k -> k == :e, keys(model.tracers))
    implicit_solver = model.timestepper.implicit_solver
    previous_clock = (; time=model.clock.time - Δt, iteration=model.clock.iteration-1)

    launch!(model.architecture, model.grid, :xyz,
            ab2_step_field!, e, Δt, χ, Gⁿe, G⁻e)

    implicit_step!(e, implicit_solver, closure,
                   model.diffusivity_fields, Val(tracer_index),
                   previous_clock, Δt)
                   
    launch!(arch, grid, :xyz, store_field_tendencies!, G⁻e, Gⁿe)

    return nothing
end

""" Calculate the right-hand-side of the subgrid scale energy equation. """
@kernel function add_tke_source_terms!(Ge, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Ge[i, j, k] += tke_source_terms(i, j, k, grid, args...)
end

@inline function tke_source_terms(i, j, k, grid,
                                  closure,
                                  next_velocities,
                                  previous_velocities,
                                  tracers,
                                  buoyancy,
                                  diffusivities)

    u⁺ = next_velocities.u
    v⁺ = next_velocities.v
    uⁿ = previous_velocities.u
    vⁿ = previous_velocities.v
    κᵘ = diffusivities.κᵘ

    return shear_production(i, j, k, grid, κᵘ, uⁿ, u⁺, vⁿ, v⁺) +
              buoyancy_flux(i, j, k, grid, closure, next_velocities, tracers, buoyancy, diffusivities) - 
                dissipation(i, j, k, grid, closure, next_velocities, tracers, buoyancy, diffusivities)
end

#=
@kernel function compute_hydrostatic_free_surface_Ge!(Ge, grid::ActiveCellsIBG, map, args)
    idx = @index(Global, Linear)
    i, j, k = active_linear_index_to_tuple(idx, map, grid)
    @inbounds Ge[i, j, k] += hydrostatic_turbulent_kinetic_energy_tendency(i, j, k, grid, args...)
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
=#


