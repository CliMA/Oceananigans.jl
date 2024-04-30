using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Grids: active_interior_map
using Oceananigans.TimeSteppers: store_field_tendencies!, ab2_step_field!, implicit_step!
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, immersed_∇_dot_qᶜ, hydrostatic_turbulent_kinetic_energy_tendency

function time_step_turbulent_kinetic_energy!(model)
    clock = model.clock
    !isfinite(clock.last_Δt) && return nothing

    tracer_name = :e
    tracer_index = findfirst(k -> k==:e, keys(model.tracers))
    Gⁿ = model.timestepper.Gⁿ.e
    G⁻ = model.timestepper.G⁻.e
    e = model.tracers.e
    closure = model.closure
    arch = model.architecture
    grid = model.grid
    Δt = model.clock.last_Δt
    χ = model.timestepper.χ

    # 1. Compute new tendency.
    e_tendency    = model.timestepper.Gⁿ.e
    e_advection   = model.advection.e
    e_forcing     = model.forcing.e
    e_immersed_bc = immersed_boundary_condition(model.tracers.e)
    active_cells_map = active_interior_map(grid)

    args = tuple(Val(tracer_index),
                 Val(:e),
                 e_advection,
                 model.closure,
                 e_immersed_bc,
                 model.buoyancy,
                 model.biogeochemistry,
                 model.velocities,
                 model.free_surface,
                 model.tracers,
                 model.diffusivity_fields,
                 model.auxiliary_fields,
                 e_forcing,
                 model.clock)

    launch!(arch, grid, :xyz,
            compute_hydrostatic_free_surface_Ge!,
            e_tendency,
            grid,
            active_cells_map,
            args;
            active_cells_map)

    # 2. Step forward
    launch!(model.architecture, model.grid, :xyz,
            ab2_step_field!, e, Δt, χ, Gⁿ, G⁻)

    implicit_step!(e,
                   model.timestepper.implicit_solver,
                   closure,
                   model.diffusivity_fields,
                   Val(tracer_index),
                   clock,
                   Δt)

    launch!(model.architecture, model.grid, :xyz,
            store_field_tendencies!,
            model.timestepper.G⁻.e,
            model.timestepper.Gⁿ.e)

    # 3. Store tendencies
    launch!(arch, grid, :xyz,
            store_field_tendencies!,
            model.timestepper.G⁻.e,
            model.timestepper.Gⁿ.e)

    return nothing
end

""" Calculate the right-hand-side of the subgrid scale energy equation. """
@kernel function compute_hydrostatic_free_surface_Ge!(Ge, grid, map, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Ge[i, j, k] = hydrostatic_turbulent_kinetic_energy_tendency(i, j, k, grid, args...)
end

#=
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


