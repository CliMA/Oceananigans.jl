using Oceananigans: fields
using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Grids: retrieve_interior_active_cells_map
using Oceananigans.BoundaryConditions: apply_x_bcs!, apply_y_bcs!, apply_z_bcs!
using Oceananigans.TimeSteppers: store_field_tendencies!, ab2_step_field!, implicit_step!
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, immersed_∇_dot_qᶜ, hydrostatic_turbulent_kinetic_energy_tendency
using CUDA

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

    Δt = model.clock.last_Δt
    Δτ = get_time_step(closure)

    if isnothing(Δτ)
        Δτ = Δt
        M = 1
    else
        M = ceil(Int, Δt / Δτ) # number of substeps
        Δτ = Δt / M
    end

    substep_turbulent_kinetic_energy!(model, Δτ, M, model.timestepper, closure, diffusivity_fields)

    return nothing
end

function substep_turbulent_kinetic_energy!(model, Δτ, M, timestepper::QuasiAdamsBashforth2TimeStepper, closure, diffusivity_fields)
    
    FT = eltype(grid)

    e = model.tracers.e
    arch = model.architecture
    grid = model.grid
    Gⁿe = timestepper.Gⁿ.e
    G⁻e = timestepper.G⁻.e

    κe = diffusivity_fields.κe
    Le = diffusivity_fields.Le
    previous_velocities = diffusivity_fields.previous_velocities
    tracer_index = findfirst(k -> k == :e, keys(model.tracers))
    implicit_solver = timestepper.implicit_solver

    for m = 1:M # substep
        if m == 1 && M != 1
            # Euler step for the first substep
            α = convert(FT, 1.0)
            β = convert(FT, 0.0)
        else
            α =   convert(FT, 1.5) + χ
            β = - convert(FT, 0.5) - χ
        end

        # Compute the linear implicit component of the RHS (diffusivities, L)
        # and step forward
        launch!(arch, grid, :xyz,
                _substep_turbulent_kinetic_energy!,
                κe, Le, grid, closure,
                model.velocities, previous_velocities, # try this soon: model.velocities, model.velocities,
                model.tracers, model.buoyancy, diffusivity_fields,
                Δτ, α, β, Gⁿe, G⁻e, nothing)

        # Good idea?
        # previous_time = model.clock.time - Δt
        # previous_iteration = model.clock.iteration - 1
        # current_time = previous_time + m * Δτ
        # previous_clock = (; time=current_time, iteration=previous_iteration)

        implicit_step!(e, implicit_solver, closure,
                    diffusivity_fields, Val(tracer_index),
                    model.clock, Δτ)
    end
end

function substep_turbulent_kinetic_energy!(model, Δτ, M, timestepper::SplitRungeKutta3TimeStepper, closure, diffusivity_fields)
    
    FT = eltype(grid)

    e = model.tracers.e
    arch = model.architecture
    grid = model.grid
    Gⁿe = timestepper.Gⁿ.e
    Ψ⁻e = timestepper.Ψ⁻.e

    κe = diffusivity_fields.κe
    Le = diffusivity_fields.Le
    previous_velocities = diffusivity_fields.previous_velocities
    tracer_index = findfirst(k -> k == :e, keys(model.tracers))
    implicit_solver = timestepper.implicit_solver

    substep_kernel!, _ = configure_kernel(arch, grid, :xyz, _substep_turbulent_kinetic_energy!)

    for m = 1:M # substep
        # Store Ψ⁻e for the next substep
        parent(Ψ⁻e) .= parent(e)

        # First RK3 substep
        α = convert(FT, 1)
        β = convert(FT, 0)

        substep_kernel!(κe, Le, grid, closure, model.velocities, previous_velocities, 
                        model.tracers, model.buoyancy, diffusivity_fields,
                        Δτ, α, β, Gⁿe, nothing, Ψ⁻e)

        implicit_step!(e, implicit_solver, closure,
                       diffusivity_fields, Val(tracer_index),
                       model.clock, Δτ)

        # Second RK3 substep
        α = timestepper.γ²
        β = timestepper.ζ²

        substep_kernel!(κe, Le, grid, closure, model.velocities, previous_velocities, 
                        model.tracers, model.buoyancy, diffusivity_fields,
                        Δτ, α, β, Gⁿe, nothing, Ψ⁻e)
        
        implicit_step!(e, implicit_solver, closure,
                       diffusivity_fields, Val(tracer_index),
                       model.clock, Δτ)

        # Third RK3 substep
        α = timestepper.γ³
        β = timestepper.ζ³

        substep_kernel!(κe, Le, grid, closure, model.velocities, previous_velocities, 
                        model.tracers, model.buoyancy, diffusivity_fields,
                        Δτ, α, β, Gⁿe, nothing, Ψ⁻e)


        implicit_step!(e, implicit_solver, closure,
                       diffusivity_fields, Val(tracer_index),
                       model.clock, Δτ)
    end
end

@kernel function _substep_turbulent_kinetic_energy!(κe, Le, grid, closure,
                                                    next_velocities, previous_velocities,
                                                    tracers, buoyancy, diffusivities,
                                                    Δτ, α, β, slow_Gⁿe, G⁻e, Ψ⁻e) 

    i, j, k = @index(Global, NTuple)

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

    # Advance TKE and store tendency
    @inbounds begin
        total_Gⁿe = slow_Gⁿe[i, j, k] + fast_Gⁿe
        advance_tke!(e, i, j, k, Δτ, α, β, total_Gⁿe, G⁻e, Ψ⁻e)
        G⁻e[i, j, k] = total_Gⁿe
    end
end

# AB2 time-stepping for the TKE equation
@inline function advance_tke!(e, i, j, k, Δτ, α, β, Gⁿe, G⁻e, ::Nothing) 
    @inbounds begin
        e[i, j, k] += Δτ *(α * Gⁿe + β * G⁻e[i, j, k])
        G⁻e[i, j, k] = Gⁿe
    end

    return nothing
end

# RK3 time-stepping for the TKE equation
@inline advance_tke!(e, i, j, k, Δτ, α, β, Gⁿe, ::Nothing, Ψ⁻e) = 
    @inbounds e[i, j, k] = β * Ψ⁻e[i, j, k] + α * (e[i, j, k] + Δτ * Gⁿe)

@inline function implicit_linear_coefficient(i, j, k, grid, ::FlavorOfCATKE{<:VITD}, K, ::Val{id}, args...) where id
    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end