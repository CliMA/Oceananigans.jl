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

tke_time_step(closure::CATKEVerticalDiffusivity) = closure.turublent_kinetic_energy_time_step
tke_time_step(closure::AbstractArray) = tke_time_step(first(closure)) # assume they are all the same

function time_step_turbulent_kinetic_energy!(model)

    e = model.tracers.e
    arch = model.architecture
    grid = model.grid
    closure = model.closure
    Gⁿe = model.timestepper.Gⁿ.e
    G⁻e = model.timestepper.G⁻.e

    diffusivity_fields = model.diffusivity_fields
    κe = diffusivity_fields.κe
    Le = diffusivity_fields.Le
    previous_velocities = diffusivity_fields.previous_velocities

    Δτ = tke_time_step(closure)
    Δt = model.clock.last_Δt
    M = ceil(Int, Δt / Δτ) # number of substeps

    for m = 1:M # substep
        
        # Euler step for the first substep
        FT = eltype(model.timestepper.χ)
        χ = m == 1 ? convert(FT, -0.5) : model.timestepper.χ

        # Compute the linear implicit component of the RHS (diffusivities, L)
        # and step forward
        launch!(arch, grid, :xyz,
                substep_turbulent_kinetic_energy!, κe, Le, grid, closure,
                model.velocities, previous_velocities, model.tracers, model.buoyancy, diffusivity_fields,
                Δτ, χ, Gⁿe, G⁻e)

        tracer_index = findfirst(k -> k == :e, keys(model.tracers))
        implicit_solver = model.timestepper.implicit_solver
        previous_clock = (; time=model.clock.time - Δt, iteration=model.clock.iteration-1)

        implicit_step!(e, implicit_solver, closure,
                       model.diffusivity_fields, Val(tracer_index),
                       previous_clock, Δτ)
    end

    return nothing
end

@kernel function substep_turbulent_kinetic_energy!(κe, Le, grid, closure,
                                                   next_velocities, previous_velocities,
                                                   tracers, buoyancy, diffusivities,
                                                   Δτ, χ, slow_Gⁿe, G⁻e)

    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Jᵇ = diffusivities.Jᵇ[i, j, 1]
        closure_ij = getclosure(i, j, closure)

        # Compute TKE diffusivity, notably omitted from calculate diffusivities.
        κe★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, Jᵇ)
        κe★ = mask_diffusivity(i, j, k, grid, κe★)
        κe[i, j, k] = κe★

        # Compute additional diagonal component of the linear TKE operator
        wb = explicit_buoyancy_flux(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, diffusivities)
        wb⁻ = min(zero(grid), wb)
        wb⁺ = max(zero(grid), wb)

        eⁱʲᵏ = tracers.e[i, j, k]
        wb⁻_e = wb⁻ / eⁱʲᵏ * (eⁱʲᵏ > 0)

        on_bottom = !inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        Cᵂϵ = closure_ij.turbulent_kinetic_energy_equation.Cᵂϵ
        e⁺ = clip(eⁱʲᵏ)
        w★ = sqrt(e⁺)
        div_Jᵉ_e = - on_bottom * Cᵂϵ * w★ / Δz

        ω = dissipation_rate(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, diffusivities)

        Le[i, j, k] = wb⁻_e - ω + div_Jᵉ_e

        # Compute fast TKE RHS
        u⁺ = next_velocities.u
        v⁺ = next_velocities.v
        uⁿ = previous_velocities.u
        vⁿ = previous_velocities.v
        κᵘ = diffusivities.κᵘ

        P = shear_production(i, j, k, grid, κᵘ, uⁿ, u⁺, vⁿ, v⁺)
        ϵ = dissipation(i, j, k, grid, closure, next_velocities, tracers, buoyancy, diffusivities)
        fast_Gⁿe = P + wb⁺ - ϵ

        # Advance TKE
        FT = eltype(χ)
        one_point_five = convert(FT, 1.5)
        oh_point_five  = convert(FT, 0.5)
        e = tracers.e

        total_Gⁿe = slow_Gⁿe[i, j, k] + fast_Gⁿe

        e[i, j, k] += convert(FT, Δτ) * ((one_point_five + χ) * total_Gⁿe - (oh_point_five + χ) * G⁻e[i, j, k])

        # Store TKE tendency
        G⁻e[i, j, k] = total_Gⁿe
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


