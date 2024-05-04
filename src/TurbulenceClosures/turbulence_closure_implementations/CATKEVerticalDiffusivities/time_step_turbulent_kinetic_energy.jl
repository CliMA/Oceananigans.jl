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

    Δt = model.clock.last_Δt
    Δτ = min(10.0, Δt)
    M = ceil(Int, Δt / Δτ)

    for m = 1:M
        Gⁿe = model.timestepper.Gⁿ.e
        e = model.tracers.e
        arch = model.architecture
        grid = model.grid
        closure = model.closure
        diffusivity_fields = model.diffusivity_fields
        κe = diffusivity_fields.κᵉ
        Le = diffusivity_fields.Lᵉ
        previous_velocities = diffusivity_fields.previous_velocities

        # Compute the linear implicit component of the RHS (diffusivities, L)
        launch!(arch, grid, :xyz,
                compute_tke_rhs!, κe, Le, Gⁿe,
                grid, closure, model.velocities, previous_velocities,
                model.tracers, model.buoyancy, diffusivity_fields)

        # 2. Step forward
        if m == 1 
            χ = -0.5
        else
            χ = model.timestepper.χ
        end

        G⁻e = model.timestepper.G⁻.e
        launch!(model.architecture, model.grid, :xyz, ab2_step_field!, e, Δτ, χ, Gⁿe, G⁻e)

        tracer_index = findfirst(k -> k == :e, keys(model.tracers))
        implicit_solver = model.timestepper.implicit_solver
        previous_clock = (; time=model.clock.time - Δt, iteration=model.clock.iteration-1)

        implicit_step!(e, implicit_solver, closure,
                       model.diffusivity_fields, Val(tracer_index),
                       previous_clock, Δτ)
                       
        launch!(arch, grid, :xyz, store_field_tendencies!, G⁻e, Gⁿe)
    end

    return nothing
end

""" Calculate the right-hand-side of the subgrid scale energy equation. """
@kernel function add_tke_source_terms!(Ge, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Ge[i, j, k] += tke_source_terms(i, j, k, grid, args...)
end

@kernel function compute_tke_source_terms!(Ge, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Ge[i, j, k] = tke_source_terms(i, j, k, grid, args...)
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

    P = shear_production(i, j, k, grid, κᵘ, uⁿ, u⁺, vⁿ, v⁺)
    wb = buoyancy_flux(i, j, k, grid, closure, next_velocities, tracers, buoyancy, diffusivities)
    ϵ = dissipation(i, j, k, grid, closure, next_velocities, tracers, buoyancy, diffusivities)

    return P + wb - ϵ
end

#@kernel function compute_linear_implicit_tke_rhs!(diffusivities, grid, closure, velocities, tracers, buoyancy)
@kernel function compute_tke_rhs!(κe, Le, Ge, grid, closure, next_velocities, previous_velocities, tracers, buoyancy, diffusivities)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Jᵇ = diffusivities.Jᵇ[i, j, 1]
        closure_ij = getclosure(i, j, closure)

        # Re-compute TKE diffusivity
        κe★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, Jᵇ)
        on_periphery = peripheral_node(i, j, k, grid, c, c, f)
        within_inactive = inactive_node(i, j, k, grid, c, c, f)
        nan = convert(eltype(grid), NaN)
        κe★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κe★))
        κe[i, j, k] = κe★

        # Compute additional diagonal component of the linear TKE operator
        wb = explicit_buoyancy_flux(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, diffusivities)
        wb⁻ = min(zero(grid), wb)
        wb⁺ = max(zero(grid), wb)

        eⁱʲᵏ = tracers.e[i, j, k]
        wb_e = wb⁻ / eⁱʲᵏ * (eⁱʲᵏ > 0)

        on_bottom = !inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        Cᵂϵ = closure_ij.turbulent_kinetic_energy_equation.Cᵂϵ
        e⁺ = clip(eⁱʲᵏ)
        w★ = sqrt(e⁺)
        div_Jᵉ_e = - on_bottom * Cᵂϵ * w★ / Δz

        ω = dissipation_rate(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, diffusivities)

        Le[i, j, k] = wb_e - ω + div_Jᵉ_e

        u⁺ = next_velocities.u
        v⁺ = next_velocities.v
        uⁿ = previous_velocities.u
        vⁿ = previous_velocities.v
        κᵘ = diffusivities.κᵘ

        P = shear_production(i, j, k, grid, κᵘ, uⁿ, u⁺, vⁿ, v⁺)
        ϵ = dissipation(i, j, k, grid, closure, next_velocities, tracers, buoyancy, diffusivities)

        Ge[i, j, k] = P + wb⁺ - ϵ
    end
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


