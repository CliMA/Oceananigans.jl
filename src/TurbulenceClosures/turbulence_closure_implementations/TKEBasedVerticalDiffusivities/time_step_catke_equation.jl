using Oceananigans: fields
using Oceananigans.Operators: σⁿ, σ⁻
using Oceananigans.Grids: bottommost_active_node
using Oceananigans.TimeSteppers: implicit_step!
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, SplitRungeKuttaTimeStepper

get_time_step(closure::CATKEVerticalDiffusivity) = closure.tke_time_step

function time_step_catke_equation!(model, ::QuasiAdamsBashforth2TimeStepper, Δt)

    # TODO: properly handle closure tuples
    if model.closure isa Tuple
        closure = first(model.closure)
        closure_fields = first(model.closure_fields)
    else
        closure = model.closure
        closure_fields = model.closure_fields
    end

    e = model.tracers.e
    arch = model.architecture
    grid = model.grid
    Gⁿe = model.timestepper.Gⁿ.e
    G⁻e = model.timestepper.G⁻.e

    κe = closure_fields.κe
    Le = closure_fields.Le
    previous_velocities = closure_fields.previous_velocities
    tracer_index = findfirst(k -> k == :e, keys(model.tracers))
    implicit_solver = model.timestepper.implicit_solver

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

        tracers = buoyancy_tracers(model)
        buoyancy = buoyancy_force(model)

        # Compute the linear implicit component of the RHS (closure_fields, L)...
        launch!(arch, grid, :xyz,
                compute_TKE_diffusivity!,
                κe, grid, closure,
                model.velocities, tracers, buoyancy, closure_fields)

        # ... and step forward.
        launch!(arch, grid, :xyz,
                _ab2_substep_turbulent_kinetic_energy!,
                Le, grid, closure,
                model.velocities, model.transport_velocities,
                tracers, buoyancy, closure_fields,
                Δτ, χ, Gⁿe, G⁻e)

        implicit_step!(e, implicit_solver, closure,
                       closure_fields, Val(tracer_index),
                       model.clock,
                       fields(model),
                       Δτ)
    end

    return nothing
end

function time_step_catke_equation!(model, ::SplitRungeKuttaTimeStepper, Δt)

    # TODO: properly handle closure tuples
    if model.closure isa Tuple
        closure = first(model.closure)
        closure_fields = first(model.closure_fields)
    else
        closure = model.closure
        closure_fields = model.closure_fields
    end

    e = model.tracers.e
    arch = model.architecture
    grid = model.grid
    Gⁿ  = model.timestepper.Gⁿ.e
    σe⁻ = model.timestepper.Ψ⁻.e

    κe = closure_fields.κe
    Le = closure_fields.Le
    previous_velocities = closure_fields.previous_velocities
    tracer_index = findfirst(k -> k == :e, keys(model.tracers))
    implicit_solver = model.timestepper.implicit_solver

    tracers = buoyancy_tracers(model)
    buoyancy = buoyancy_force(model)

    # Compute the linear implicit component of the RHS (closure_fields, L)...
    launch!(arch, grid, :xyz,
            compute_TKE_diffusivity!,
            κe, grid, closure,
            model.velocities, tracers, buoyancy, closure_fields)

    launch!(arch, grid, :xyz,
            _rk_substep_turbulent_kinetic_energy!,
            Le, σe⁻, grid, closure,
            model.velocities, model.transport_velocities,
            tracers, buoyancy, closure_fields,
            Δt, Gⁿ)

    implicit_step!(e, implicit_solver, closure,
                   closure_fields, Val(tracer_index),
                   model.clock,
                   fields(model),
                   Δτ)

    return nothing
end

const c = Center()

@kernel function compute_TKE_diffusivity!(κe, grid, closure,
                                          next_velocities, tracers, buoyancy, closure_fields)
    i, j, k = @index(Global, NTuple)

    # Compute TKE diffusivity.
    closure_ij = getclosure(i, j, closure)
    Jᵇ = closure_fields.Jᵇ
    κe★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, Jᵇ)
    κe★ = mask_diffusivity(i, j, k, grid, κe★)
    @inbounds κe[i, j, k] = κe★
end

@inline function fast_tke_tendency(i, j, k, grid, Le, closure,
                                   next_velocities, previous_velocities,
                                   tracers, buoyancy, closure_fields)

    e = tracers.e
    closure_ij = getclosure(i, j, closure)

    # Compute additional diagonal component of the linear TKE operator
    wb = explicit_buoyancy_flux(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, closure_fields)
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

    on_bottom = bottommost_active_node(i, j, k, grid, c, c, c)
    active = !inactive_cell(i, j, k, grid)
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    Cᵂϵ = closure_ij.turbulent_kinetic_energy_equation.Cᵂϵ
    e⁺ = clip(eⁱʲᵏ)
    w★ = sqrt(e⁺)
    div_Jᵉ_e = - on_bottom * Cᵂϵ * w★ / Δz

    # Implicit TKE dissipation
    ω = dissipation_rate(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, closure_fields)

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
    κu = closure_fields.κu

    # TODO: correctly handle closure / diffusivity tuples
    # TODO: the shear_production is actually a slow term so we _could_ precompute.
    P = shear_production(i, j, k, grid, κu, uⁿ, u⁺, vⁿ, v⁺)
    ϵ = dissipation(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, closure_fields)
    return P + wb⁺ - ϵ
end

@kernel function _ab2_substep_turbulent_kinetic_energy!(Le, grid, closure,
                                                        next_velocities, previous_velocities,
                                                        tracers, buoyancy, closure_fields,
                                                        Δτ, χ, slow_Gⁿe, G⁻e)

    i, j, k = @index(Global, NTuple)

    fast_Gⁿe = fast_tke_tendency(i, j, k, grid, Le, closure,
                                 next_velocities, previous_velocities,
                                 tracers, buoyancy, closure_fields)

    # Advance TKE and store tendency
    FT = eltype(χ)
    Δτ = convert(FT, Δτ)
    e  = tracers.e

    # See below.
    α = convert(FT, 1.5) + χ
    β = convert(FT, 0.5) + χ

    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    σᶜᶜ⁻ = σ⁻(i, j, k, grid, Center(), Center(), Center())
    active = !inactive_cell(i, j, k, grid)

    @inbounds begin
        total_Gⁿe = slow_Gⁿe[i, j, k] + fast_Gⁿe * σᶜᶜⁿ
        e[i, j, k] += Δτ * (α * total_Gⁿe - β * G⁻e[i, j, k]) * active / σᶜᶜⁿ
        G⁻e[i, j, k] = total_Gⁿe * active
    end
end

@kernel function _rk_substep_turbulent_kinetic_energy!(Le, σe⁻, grid, closure,
                                                       next_velocities, previous_velocities,
                                                       tracers, buoyancy, closure_fields,
                                                       Δt, slow_Gⁿe)

    i, j, k = @index(Global, NTuple)

    e = tracers.e

    fast_Gⁿe = fast_tke_tendency(i, j, k, grid, Le, closure,
                                 next_velocities, previous_velocities,
                                 tracers, buoyancy, closure_fields)

    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    active = !inactive_cell(i, j, k, grid)

    @inbounds begin
        total_Gⁿ = slow_Gⁿe[i, j, k] + fast_Gⁿe * σᶜᶜⁿ
        e[i, j, k] = (σe⁻[i, j, k] + Δt * total_Gⁿ * active) / σᶜᶜⁿ
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

function tracer_tendency_kernel_function(model::HFSM, ::Val{:e}, closures::Tuple, closure_fields::Tuple)
    catke_index = findfirst(c -> c isa FlavorOfCATKE, closures)

    if isnothing(catke_index)
        return compute_hydrostatic_free_surface_Gc!, closures, closure_fields
    else
        catke_closure = closures[catke_index]
        catke_closure_fields = closure_fields[catke_index]
        return compute_hydrostatic_free_surface_Ge!, catke_closure, catke_closure_fields
    end
end

@inline function top_tracer_boundary_conditions(grid, tracers)
    names = propertynames(tracers)
    values = Tuple(tracers[c].boundary_conditions.top for c in names)

    # Some shenanigans for type stability?
    return NamedTuple{tuple(names...)}(tuple(values...))
end
=#
