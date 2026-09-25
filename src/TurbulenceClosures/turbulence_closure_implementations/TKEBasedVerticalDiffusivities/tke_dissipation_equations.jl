using Oceananigans: fields
using Oceananigans.Fields: znode
using Oceananigans.TimeSteppers: implicit_step!

Base.@kwdef struct TKEDissipationEquations{FT}
    Cᵋϵ :: FT = 1.92
    Cᴾϵ :: FT = 1.44
    Cᵇϵ⁺ :: FT = -0.65
    Cᵇϵ⁻ :: FT = -0.65
    Cᵂu★ :: FT = 0.0
    CᵂwΔ :: FT = 0.0
    Cᵂα  :: FT = 0.11 # Charnock parameter
    gravitational_acceleration :: FT = 9.8065
    minimum_roughness_length :: FT = 1e-4
end

get_time_step(closure::TKEDissipationVerticalDiffusivity) = closure.tke_dissipation_time_step

function time_step_tke_dissipation_equations!(model, Δt)

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

    closure_fields = model.closure_fields
    κe = closure_fields.κe
    κϵ = closure_fields.κϵ
    Le = closure_fields.Le
    Lϵ = closure_fields.Lϵ
    previous_velocities = closure_fields.previous_velocities
    e_index = findfirst(k -> k == :e, keys(model.tracers))
    ϵ_index = findfirst(k -> k == :ϵ, keys(model.tracers))
    implicit_solver = model.timestepper.implicit_solver
    active_cells_map = get_active_cells_map(grid, Val(:xyz))

    FT = eltype(model.tracers.e)
    Δτ = get_time_step(closure)

    if isnothing(Δτ)
        Δτ = Δt
        M = 1
    else
        M = ceil(Int, Δt / Δτ) # number of substeps
        Δτ = Δt / M
    end

    for m = 1:M # substep
        if m == 1 && M != 1
            χ = convert(FT, -0.5) # Euler step for the first substep
        else
            χ = model.timestepper.χ
        end

        launch!(arch, grid, :xyz,
                compute_tke_dissipation_closure_fields!,
                κe, κϵ,
                grid, closure,
                model.velocities, model.tracers, buoyancy_force(model);
                active_cells_map)

        # Compute the linear implicit component of the RHS (closure_fields, L)
        # and step forward
        launch!(arch, grid, :xyz,
                substep_tke_dissipation!,
                Le, Lϵ,
                grid, closure,
                model.velocities, previous_velocities,
                model.tracers, buoyancy_force(model), closure_fields,
                Δτ, χ, Gⁿe, G⁻e, Gⁿϵ, G⁻ϵ;
                active_cells_map)

        implicit_step!(e, implicit_solver, closure,
                       model.closure_fields, Val(e_index),
                       model.clock,
                       fields(model),
                       Δτ)

        implicit_step!(ϵ, implicit_solver, closure,
                       model.closure_fields, Val(ϵ_index),
                       model.clock,
                       fields(model),
                       Δτ)
    end

    return nothing
end

# Compute TKE and dissipation closure_fields
@kernel function compute_tke_dissipation_closure_fields!(κe, κϵ, grid, closure,
                                                        velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    κe★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)
    κϵ★ = κϵᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy)
    κe★ = mask_diffusivity(i, j, k, grid, κe★)
    κϵ★ = mask_diffusivity(i, j, k, grid, κϵ★)
    @inbounds κe[i, j, k] = κe★
    @inbounds κϵ[i, j, k] = κϵ★
end

@kernel function substep_tke_dissipation!(Le, Lϵ,
                                          grid, closure,
                                          next_velocities, previous_velocities,
                                          tracers, buoyancy, closure_fields,
                                          Δτ, χ, slow_Gⁿe, G⁻e, slow_Gⁿϵ, G⁻ϵ)

    i, j, k = @index(Global, NTuple)

    e = tracers.e
    ϵ = tracers.ϵ
    closure_ij = getclosure(i, j, closure)

    # Compute TKE and dissipation tendencies
    ϵ★ = dissipationᶜᶜᶜ(i, j, k, grid, closure_ij, tracers, buoyancy)
    e★ = turbulent_kinetic_energyᶜᶜᶜ(i, j, k, grid, closure_ij, tracers)
    eⁱʲᵏ = @inbounds e[i, j, k]
    ϵⁱʲᵏ = @inbounds ϵ[i, j, k]

    # Different destruction time-scales for TKE vs dissipation for numerical reasons
    ω★  = ϵ★ / e★ # target / physical dissipation time scale
    ωe⁻ = 1 / closure_ij.negative_tke_damping_time_scale  # frequency = 1/timescale
    ωe  = ifelse(eⁱʲᵏ < 0, ωe⁻, ω★)
    ωϵ  = ϵⁱʲᵏ / e★

    # Compute additional diagonal component of the linear TKE operator
    wb = explicit_buoyancy_flux(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, closure_fields)

    # Patankar trick for TKE equation
    wb⁻ = min(wb, zero(grid))
    wb⁺ = max(wb, zero(grid))

    eᵐⁱⁿ = closure_ij.minimum_tke
    wb⁻_e = wb⁻ / eⁱʲᵏ * (eⁱʲᵏ > eᵐⁱⁿ)

    # Patankar trick for ϵ-equation
    Cᵋϵ = closure_ij.tke_dissipation_equations.Cᵋϵ
    Cᵇϵ⁺ = closure_ij.tke_dissipation_equations.Cᵇϵ⁺
    Cᵇϵ⁻ = closure_ij.tke_dissipation_equations.Cᵇϵ⁻

    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    Cᵇϵ = ifelse(N² ≥ 0, Cᵇϵ⁺, Cᵇϵ⁻)

    Cᵇϵ_wb⁻ = min(Cᵇϵ * wb, zero(grid))
    Cᵇϵ_wb⁺ = max(Cᵇϵ * wb, zero(grid))

    # ∂t e = Lⁱ e + ⋯,
    @inbounds Le[i, j, k] = wb⁻_e - ωe
    @inbounds Lϵ[i, j, k] = Cᵇϵ_wb⁻ / e★ - Cᵋϵ * ωϵ

    # Compute fast TKE and dissipation RHSs
    u⁺ = next_velocities.u
    v⁺ = next_velocities.v
    uⁿ = previous_velocities.u
    vⁿ = previous_velocities.v
    κu = closure_fields.κu
    Cᴾϵ = closure_ij.tke_dissipation_equations.Cᴾϵ

    # TODO: correctly handle closure / diffusivity tuples
    # TODO: the shear_production is actually a slow term so we _could_ precompute.
    P = shear_production(i, j, k, grid, κu, uⁿ, u⁺, vⁿ, v⁺)

    @inbounds begin
        fast_Gⁿe = P + wb⁺                  # - ϵ (no implicit time stepping for now)
        fast_Gⁿϵ = ωϵ * (Cᴾϵ * P + Cᵇϵ_wb⁺)
    end

    # Advance TKE and store tendency
    FT = eltype(e)
    χ = convert(FT, χ)

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

    return - Cᵂu★ * u★^3 #- CᵂwΔ * wΔ³
end

@inline function top_dissipation_flux(i, j, grid, clock, fields, parameters, closure::FlavorOfTD, buoyancy)
    closure = getclosure(i, j, closure)

    top_tracer_bcs = parameters.top_tracer_boundary_conditions
    top_velocity_bcs = parameters.top_velocity_boundary_conditions
    tke_dissipation_parameters = closure.tke_dissipation_equations

    return _top_dissipation_flux(i, j, grid, clock, fields, tke_dissipation_parameters, closure,
                                 buoyancy, top_tracer_bcs, top_velocity_bcs)
end

@inline function _top_dissipation_flux(i, j, grid, clock, fields, parameters::TKEDissipationEquations,
                                       closure::TDVD, buoyancy, top_tracer_bcs, top_velocity_bcs)

    𝕊u₀ = closure.stability_functions.𝕊u₀
    σϵ = closure.stability_functions.Cσϵ

    u★ = friction_velocity(i, j, grid, clock, fields, top_velocity_bcs)
    α = parameters.Cᵂα
    g = parameters.gravitational_acceleration
    ℓ_charnock = α * u★^2 / g

    ℓmin = parameters.minimum_roughness_length
    ℓᵣ = max(ℓmin, ℓ_charnock)

    k = grid.Nz
    e★ = turbulent_kinetic_energyᶜᶜᶜ(i, j, k, grid, closure, fields)
    z = znode(i, j, k, grid, c, c, c)
    d = - z

    return - 𝕊u₀^4 / σϵ * e★^2 / (d + ℓᵣ)
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
    top_dissipation_bc = FluxBoundaryCondition(top_dissipation_flux, discrete_form=true, parameters=parameters)

    if :e ∈ keys(user_bcs)
        e_bcs = user_bcs[:e]

        tke_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()),
                                          top = top_tke_bc,
                                          bottom = e_bcs.bottom,
                                          north = e_bcs.north,
                                          south = e_bcs.south,
                                          east = e_bcs.east,
                                          west = e_bcs.west)
    else
        tke_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()), top=top_tke_bc)
    end

    if :ϵ ∈ keys(user_bcs)
        ϵ_bcs = user_bcs[:ϵ]

        dissipation_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()),
                                                  top = top_dissipation_bc,
                                                  bottom = ϵ_bcs.bottom,
                                                  north = ϵ_bcs.north,
                                                  south = ϵ_bcs.south,
                                                  east = ϵ_bcs.east,
                                                  west = ϵ_bcs.west)
    else
        dissipation_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()), top=top_dissipation_bc)
    end

    new_boundary_conditions = merge(user_bcs, (e=tke_bcs, ϵ=dissipation_bcs))

    return new_boundary_conditions
end
