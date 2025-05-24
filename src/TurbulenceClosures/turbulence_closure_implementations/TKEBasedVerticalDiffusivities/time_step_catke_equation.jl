using Oceananigans: fields
using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Grids: get_active_cells_map
using Oceananigans.BoundaryConditions: apply_x_bcs!, apply_y_bcs!, apply_z_bcs!
using Oceananigans.TimeSteppers: ab2_step_field!, implicit_step!
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, immersed_∇_dot_qᶜ, hydrostatic_turbulent_kinetic_energy_tendency

get_time_step(closure::CATKEVerticalDiffusivity) = closure.tke_time_step

function time_step_catke_equation!(model)

    # TODO: properly handle closure tuples
    if model.closure isa Tuple
        closure = first(model.closure)
        diffusivity_fields = first(model.diffusivity_fields)
    else
        closure = model.closure
        diffusivity_fields = model.diffusivity_fields
    end

    e = model.tracers.e
    arch = model.architecture
    grid = model.grid
    Gⁿe = model.timestepper.Gⁿ.e
    G⁻e = model.timestepper.G⁻.e

    κe = diffusivity_fields.κe
    Le = diffusivity_fields.Le
    previous_velocities = diffusivity_fields.previous_velocities
    tracer_index = findfirst(k -> k == :e, keys(model.tracers))
    implicit_solver = model.timestepper.implicit_solver

    Δt = model.clock.last_Δt
    Δτ = get_time_step(closure)

    if isnothing(Δτ)
        Δτ = Δt
        M = 1
    else
        M = ceil(Int, Δt / Δτ) # number of substeps
        Δτ = Δt / M
    end

    FT = eltype(grid)

    for m = 1:M # substep
        if m == 1 && M != 1
            χ = convert(FT, -0.5) # Euler step for the first substep
        else
            χ = model.timestepper.χ
        end

        # Compute the linear implicit component of the RHS (diffusivities, L)...
        launch!(arch, grid, :xyz,
                compute_TKE_diffusivity!,
                κe, grid, closure,
                model.velocities, model.tracers, model.buoyancy, diffusivity_fields)
                
        # ... and step forward.
        launch!(arch, grid, :xyz,
                substep_turbulent_kinetic_energy!,
                Le, grid, closure,
                model.velocities, previous_velocities, # try this soon: model.velocities, model.velocities,
                model.tracers, model.buoyancy, diffusivity_fields,
                Δτ, χ, Gⁿe, G⁻e)

        # Good idea?
        # previous_time = model.clock.time - Δt
        # previous_iteration = model.clock.iteration - 1
        # current_time = previous_time + m * Δτ
        # previous_clock = (; time=current_time, iteration=previous_iteration)

        implicit_step!(e, implicit_solver, closure,
                       diffusivity_fields, Val(tracer_index),
                       model.clock, Δτ)
    end

    return nothing
end

@kernel function compute_TKE_diffusivity!(κe, grid, closure,
                                          next_velocities, tracers, buoyancy, diffusivities)
    i, j, k = @index(Global, NTuple)

    # Compute TKE diffusivity.
    closure_ij = getclosure(i, j, closure)
    Jᵇ = diffusivities.Jᵇ
    κe★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, Jᵇ)
    κe★ = mask_diffusivity(i, j, k, grid, κe★)
    @inbounds κe[i, j, k] = κe★
end

@kernel function substep_turbulent_kinetic_energy!(Le, grid, closure,
                                                   next_velocities, previous_velocities,
                                                   tracers, buoyancy, diffusivities,
                                                   Δτ, χ, slow_Gⁿe, G⁻e)

    i, j, k = @index(Global, NTuple)

    e = tracers.e
    closure_ij = getclosure(i, j, closure)

    # Compute additional diagonal component of the linear TKE operator
    wb = explicit_buoyancy_flux(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, diffusivities)
    wb⁻ = min(zero(grid), wb)
    wb⁺ = max(zero(grid), wb)

    eⁱʲᵏ = @inbounds e[i, j, k]
    eᵐⁱⁿ = closure_ij.minimum_tke
    wb⁻_e = wb⁻ / eⁱʲᵏ * (eⁱʲᵏ > eᵐⁱⁿ)

    # Treat the divergence of TKE flux at solid bottoms implicitly.
    # This will damp TKE near boundaries. The bottom-localized TKE flux may be written
    #
    #       ∂t e = - δ(z + h) ∇ ⋅ Jᵉ + ⋯
    #       ∂t e = + δ(z + h) Jᵉ / Δz + ⋯
    #
    # where δ(z + h) is a δ-function that is 0 everywhere except adjacent to the bottom boundary
    # at $z = -h$ and Δz is the grid spacing at the bottom
    #
    # Thus if
    #
    #       Jᵉ ≡ - Cᵂϵ * √e³
    #          = - (Cᵂϵ * √e) e
    #
    # Then the contribution of Jᵉ to the implicit flux is
    #
    #       Lᵂ = - Cᵂϵ * √e / Δz.

    on_bottom = !inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)
    active = !inactive_cell(i, j, k, grid)
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    Cᵂϵ = closure_ij.turbulent_kinetic_energy_equation.Cᵂϵ
    e⁺ = clip(eⁱʲᵏ)
    w★ = sqrt(e⁺)
    div_Jᵉ_e = - on_bottom * Cᵂϵ * w★ / Δz

    # Implicit TKE dissipation
    ω = dissipation_rate(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, diffusivities)

    # The interior contributions to the linear implicit term `L` are defined via
    #
    #       ∂t e = Lⁱ e + ⋯,
    #
    # So
    #
    #       Lⁱ e = wb - ϵ
    #            = (wb / e - ω) e,
    #               ↖--------↗
    #                  = Lⁱ
    #
    # where ω = ϵ / e ∼ √e / ℓ.

    @inbounds Le[i, j, k] = (wb⁻_e - ω + div_Jᵉ_e) * active

    # Compute fast TKE RHS
    u⁺ = next_velocities.u
    v⁺ = next_velocities.v
    uⁿ = previous_velocities.u
    vⁿ = previous_velocities.v
    κu = diffusivities.κu

    # TODO: correctly handle closure / diffusivity tuples
    # TODO: the shear_production is actually a slow term so we _could_ precompute.
    P = shear_production(i, j, k, grid, κu, uⁿ, u⁺, vⁿ, v⁺)
    ϵ = dissipation(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, diffusivities)
    fast_Gⁿe = P + wb⁺ - ϵ

    # Advance TKE and store tendency
    FT = eltype(χ)
    Δτ = convert(FT, Δτ)

    # See below.
    α = convert(FT, 1.5) + χ
    β = convert(FT, 0.5) + χ

    @inbounds begin
        total_Gⁿe = slow_Gⁿe[i, j, k] + fast_Gⁿe
        e[i, j, k] += Δτ * (α * total_Gⁿe - β * G⁻e[i, j, k]) * active
        G⁻e[i, j, k] = total_Gⁿe * active
    end
end

@inline function implicit_linear_coefficient(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, K, ::Val{id}, args...) where id
    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end

#=
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: FlavorOfCATKE

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
