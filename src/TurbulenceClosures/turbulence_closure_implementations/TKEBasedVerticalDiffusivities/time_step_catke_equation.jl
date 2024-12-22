using Oceananigans.TimeSteppers: implicit_step!, QuasiAdamsBashforth2TimeStepper, SplitRungeKutta3TimeStepper

get_time_step(closure::FlavorOfCATKEWithSubsteps, Δt) = closure.tke_time_step
get_time_step(::FlavorOfCATKEWithoutSubsteps, Δt) = Δt

function time_step_catke_equation!(model)

    # TODO: properly handle closure tuples
    if model.closure isa Tuple
        closure = first(model.closure)
        diffusivity_fields = first(model.diffusivity_fields)
    else
        closure = model.closure
        diffusivity_fields = model.diffusivity_fields
    end

    Δt = model.clock.last_Δt
    
    if Δt == 0 || !isfinite(Δt) # Not a valid time step
        return nothing
    end

    Δτ = get_time_step(closure, Δt)
    M  = ceil(Int, Δt / Δτ) # Number of substeps

    grid = model.grid
    e   = model.tracers.e
    Gⁿe = timestepper.Gⁿ.e
    FT  = eltype(grid)

    κe = diffusivity_fields.κe
    Le = diffusivity_fields.Le
    previous_velocities = diffusivity_fields.previous_velocities
    tracer_index = findfirst(k -> k == :e, keys(model.tracers))
    implicit_solver = timestepper.implicit_solver

    active_cells_map  = retrieve_interior_active_cells_map(model.grid, Val(:interior))

    if M == 1 # Just add the fast terms. Substepping will be done with the other tracers
        launch!(architecture(grid), grid, :xyz,
                _compute_tke_tendency!,
                Gⁿe, κe, Le, grid, closure,
                model.velocities, previous_velocities, 
                model.tracers, model.buoyancy, diffusivity_fields,
                active_cells_map)

    else # Substep using an AB2 scheme for the fast evolution terms
        # Euler step for the first substep
        α = convert(FT, 1.0)
        β = convert(FT, 0.0)

        G⁻e = timestepper.G⁻.e

        for m = 1:M # substep
            # Compute the linear implicit component of the RHS (diffusivities, L)
            # and step forward
            launch!(architecture(grid), grid, :xyz,
                    _substep_turbulent_kinetic_energy!,
                    κe, Le, grid, closure,
                    model.velocities, previous_velocities, # try this soon: model.velocities, model.velocities,
                    model.tracers, model.buoyancy, diffusivity_fields,
                    Δτ, α, β, Gⁿe, G⁻e;
                    active_cells_map)

            # Good idea?
            # previous_time = model.clock.time - Δt
            # previous_iteration = model.clock.iteration - 1
            # current_time = previous_time + m * Δτ
            # previous_clock = (; time=current_time, iteration=previous_iteration)
            implicit_step!(e, implicit_solver, closure,
                        diffusivity_fields, Val(tracer_index),
                        model.clock, Δτ)

            # Update the time stepping coefficients
            α =   convert(FT, 1.5) + timestepper.χ
            β = - convert(FT, 0.5) - timestepper.χ
        end
    end
end

@kernel function _compute_tke_tendency!(Gⁿe, κe, Le, grid, closure, next_velocities, previous_velocities,
                                        tracers, buoyancy, diffusivities)

    i, j, k = @index(Global, NTuple)
    @inbounds Gⁿe[i, j, k] = compute_tke_tendency(i, j, k, grid, κe, Le, closure, next_velocities, previous_velocities,
                                                  tracers, buoyancy, diffusivities, Gⁿe)
end

@kernel function _substep_turbulent_kinetic_energy!(κe, Le, grid, closure,
                                                    next_velocities, previous_velocities,
                                                    tracers, buoyancy, diffusivities,
                                                    Δτ, α, β, slow_Gⁿe, G⁻e) 

    i, j, k = @index(Global, NTuple)

    # Compute total tendency
    @inbounds Gⁿe = compute_tke_tendency(i, j, k, grid, κe, Le, closure, next_velocities, previous_velocities,
                                         tracers, buoyancy, diffusivities, slow_Gⁿe)

    # Advance TKE and store tendency
    @inbounds begin
        e[i, j, k] += Δτ * (α * Gⁿe + β * G⁻e[i, j, k])
        G⁻e[i, j, k] = Gⁿe
    end
end

@inline function compute_tke_tendency(i, j, k, grid, κe, Le, closure, next_velocities, previous_velocities,
                                      tracers, buoyancy, diffusivities, slow_Gⁿe)

    Jᵇ = diffusivities.Jᵇ
    e = tracers.e
    closure_ij = getclosure(i, j, closure)

    # Compute TKE diffusivity.
    κe★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, Jᵇ)
    κe★ = mask_diffusivity(i, j, k, grid, κe★)
    @inbounds κe[i, j, k] = κe★

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

    @inbounds Le[i, j, k] = wb⁻_e - ω + div_Jᵉ_e

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

    return fast_Gⁿe + slow_Gⁿe[i, j, k]
end

@inline function implicit_linear_coefficient(i, j, k, grid, ::FlavorOfCATKE{<:VITD}, K, ::Val{id}, args...) where id
    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end