using Oceananigans: fields
using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Grids: active_interior_map
using Oceananigans.BoundaryConditions: apply_x_bcs!, apply_y_bcs!, apply_z_bcs!
using Oceananigans.TimeSteppers: store_field_tendencies!, ab2_step_field!, implicit_step!
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, immersed_∇_dot_qᶜ, hydrostatic_turbulent_kinetic_energy_tendency
using CUDA

Base.@kwdef struct TKEDissipationEquations{FT}
    Cᵋϵ :: FT = 1.92
    Cᴾϵ :: FT = 1.44
    Cᵇϵ :: FT = -0.65
    Cᵂu★ :: FT = 1.0
    CᵂwΔ :: FT = 1.0
end

get_time_step(closure::TKEDissipationVerticalDiffusivity) = closure.tke_dissipation_time_step

function time_step_tke_dissipation_equations!(model)

    # TODO: properly handle closure tuples
    closure = model.closure

    e = model.tracers.e
    ϵ = model.tracers.ϵ
    arch = model.architecture
    grid = model.grid
    Gⁿe = model.timestepper.Gⁿ.e
    G⁻e = model.timestepper.G⁻.e
    Gⁿϵ = model.timestepper.Gⁿ.ϵ
    G⁻ϵ = model.timestepper.G⁻.ϵ

    diffusivity_fields = model.diffusivity_fields
    κe = diffusivity_fields.κe
    κϵ = diffusivity_fields.κϵ
    Le = diffusivity_fields.Le
    Lϵ = diffusivity_fields.Lϵ
    previous_velocities = diffusivity_fields.previous_velocities
    e_index = findfirst(k -> k == :e, keys(model.tracers))
    ϵ_index = findfirst(k -> k == :ϵ, keys(model.tracers))
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

        # Compute the linear implicit component of the RHS (diffusivities, L)
        # and step forward
        launch!(arch, grid, :xyz,
                substep_tke_dissipation!,
                κe, κϵ, Le, Lϵ,
                grid, closure,
                model.velocities, previous_velocities, # try this soon: model.velocities, model.velocities,
                model.tracers, model.buoyancy, diffusivity_fields,
                Δτ, χ, Gⁿe, G⁻e, Gⁿϵ, G⁻ϵ)

        # Good idea?
        # previous_time = model.clock.time - Δt
        # previous_iteration = model.clock.iteration - 1
        # current_time = previous_time + m * Δτ
        # previous_clock = (; time=current_time, iteration=previous_iteration)

        implicit_step!(e, implicit_solver, closure,
                       model.diffusivity_fields, Val(e_index),
                       model.clock, Δτ)

        implicit_step!(ϵ, implicit_solver, closure,
                       model.diffusivity_fields, Val(ϵ_index),
                       model.clock, Δτ)
    end

    return nothing
end

@kernel function substep_tke_dissipation!(κe, κϵ, Le, Lϵ,
                                          grid, closure,
                                          next_velocities, previous_velocities,
                                          tracers, buoyancy, diffusivities,
                                          Δτ, χ, slow_Gⁿe, G⁻e, slow_Gⁿϵ, G⁻ϵ)

    i, j, k = @index(Global, NTuple)

    e = tracers.e
    ϵ = tracers.ϵ

    closure_ij = getclosure(i, j, closure)

    # Compute TKE diffusivity.
    κe★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy)
    κϵ★ = κϵᶜᶜᶠ(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy)

    κe★ = mask_diffusivity(i, j, k, grid, κe★)
    κϵ★ = mask_diffusivity(i, j, k, grid, κϵ★)

    @inbounds κe[i, j, k] = κe★
    @inbounds κϵ[i, j, k] = κϵ★

    # Compute TKE and dissipation tendencies
    ϵ★ = dissipationᶜᶜᶜ(i, j, k, grid, closure, tracers, buoyancy)
    e★ = turbulent_kinetic_energyᶜᶜᶜ(i, j, k, grid, closure, tracers)
    eⁱʲᵏ = @inbounds e[i, j, k]
    ϵⁱʲᵏ = @inbounds ϵ[i, j, k]

    # Different destruction time-scales for TKE vs dissipation for numerical reasons
    ω★  = ϵ★ / e★ # target / physical dissipation time scale
    ωe⁻ = closure.negative_tke_damping_time_scale
    ωe  = ifelse(eⁱʲᵏ < 0, ωe⁻, ω★)
    ωϵ  = ϵⁱʲᵏ / e★

    # Compute additional diagonal component of the linear TKE operator
    wb = explicit_buoyancy_flux(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, diffusivities)
    wb⁻ = min(zero(grid), wb)
    wb⁺ = max(zero(grid), wb)

    eᵐⁱⁿ = closure_ij.minimum_tke
    wb⁻_e = wb⁻ / eⁱʲᵏ * (eⁱʲᵏ > eᵐⁱⁿ)

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

    Cᵋϵ = closure_ij.tke_dissipation_equations.Cᵋϵ
    Cᵇϵ = closure_ij.tke_dissipation_equations.Cᵇϵ

    @inbounds Le[i, j, k] = wb⁻_e - ωe
    @inbounds Lϵ[i, j, k] = Cᵇϵ * wb⁻_e - Cᵋϵ * ωϵ

    # Compute fast TKE and dissipation RHSs
    u⁺ = next_velocities.u
    v⁺ = next_velocities.v
    uⁿ = previous_velocities.u
    vⁿ = previous_velocities.v
    κu = diffusivities.κu
    Cᴾϵ = closure_ij.tke_dissipation_equations.Cᴾϵ

    # TODO: correctly handle closure / diffusivity tuples
    # TODO: the shear_production is actually a slow term so we _could_ precompute.
    P = shear_production(i, j, k, grid, κu, uⁿ, u⁺, vⁿ, v⁺)

    @inbounds begin
        fast_Gⁿe = P + wb⁺                          # - ϵ (no implicit time stepping for now)
        fast_Gⁿϵ = ω★ * (Cᴾϵ * P + Cᵇϵ * wb⁺) # - ϵ
    end

    # Advance TKE and store tendency
    FT = eltype(χ)
    Δτ = convert(FT, Δτ)

    # See below.
    α = convert(FT, 1.5) + χ
    β = convert(FT, 0.5) + χ

    @inbounds begin
        total_Gⁿe = slow_Gⁿe[i, j, k] + fast_Gⁿe
        total_Gⁿϵ = slow_Gⁿϵ[i, j, k] + fast_Gⁿϵ

        e[i, j, k] += Δτ * (α * total_Gⁿe - β * G⁻e[i, j, k])
        ϵ[i, j, k] += Δτ * (α * total_Gⁿϵ - β * G⁻ϵ[i, j, k])

        G⁻e[i, j, k] = total_Gⁿe
        G⁻ϵ[i, j, k] = total_Gⁿϵ
    end
end

@inline function implicit_linear_coefficient(i, j, k, grid, closure::FlavorOfTD{<:VITD}, K, ::Val{id}, args...) where id
    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end

#####
##### TKE top boundary condition
#####

@inline function top_tke_flux(i, j, grid, clock, fields, parameters, closure::FlavorOfTD, buoyancy)
    closure = getclosure(i, j, closure)

    top_tracer_bcs = parameters.top_tracer_boundary_conditions
    top_velocity_bcs = parameters.top_velocity_boundary_conditions
    tke_dissipation_parameters = closure.tke_dissipation_equations

    return _top_tke_flux(i, j, grid, clock, fields, tke_dissipation_parameters, closure,
                         buoyancy, top_tracer_bcs, top_velocity_bcs)
end

@inline function _top_tke_flux(i, j, grid, clock, fields,
                               parameters::TKEDissipationEquations, closure::TDVD,
                               buoyancy, top_tracer_bcs, top_velocity_bcs)

    wΔ³ = top_convective_turbulent_velocity_cubed(i, j, grid, clock, fields, buoyancy, top_tracer_bcs)
    u★ = friction_velocity(i, j, grid, clock, fields, top_velocity_bcs)

    Cᵂu★ = parameters.Cᵂu★
    CᵂwΔ = parameters.CᵂwΔ

    return - Cᵂu★ * u★^3 - CᵂwΔ * wΔ³
end

#####
##### Utilities for model constructors
#####

""" Add TKE boundary conditions specific to `TKEDissipationVerticalDiffusivity`. """
function add_closure_specific_boundary_conditions(closure::FlavorOfTD,
                                                  user_bcs,
                                                  grid,
                                                  tracer_names,
                                                  buoyancy)

    top_tracer_bcs = top_tracer_boundary_conditions(grid, tracer_names, user_bcs)
    top_velocity_bcs = top_velocity_boundary_conditions(grid, user_bcs)

    parameters = TKETopBoundaryConditionParameters(top_tracer_bcs, top_velocity_bcs)

    top_tke_bc = FluxBoundaryCondition(top_tke_flux, discrete_form=true, parameters=parameters)

    if :e ∈ keys(user_bcs)
        e_bcs = user_bcs[:e]
        
        tke_bcs = FieldBoundaryConditions(grid, (Center, Center, Center),
                                          top = top_tke_bc,
                                          bottom = e_bcs.bottom,
                                          north = e_bcs.north,
                                          south = e_bcs.south,
                                          east = e_bcs.east,
                                          west = e_bcs.west)
    else
        tke_bcs = FieldBoundaryConditions(grid, (Center, Center, Center), top=top_tke_bc)
    end

    new_boundary_conditions = merge(user_bcs, (; e = tke_bcs))

    return new_boundary_conditions
end

