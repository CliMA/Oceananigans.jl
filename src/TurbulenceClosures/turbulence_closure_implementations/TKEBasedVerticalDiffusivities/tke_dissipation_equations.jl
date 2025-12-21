using Oceananigans: fields
using Oceananigans.Fields: znode
using Oceananigans.TimeSteppers: implicit_step!

#####
##### TKE-dissipation equation coefficients
#####

Base.@kwdef struct TKEDissipationEquations{FT}
    Cᵋϵ :: FT = 1.92
    Cᴾϵ :: FT = 1.44
    Cᵇϵ⁺ :: FT = -0.65
    Cᵇϵ⁻ :: FT = -0.65
end

#####
##### Boundary condition types for TKE and dissipation
#####

"""
    SurfaceTKEBoundaryCondition{FT}

Parameters for TKE injection at a surface (solid or free).
The TKE flux is computed as:

    Qₑ = - Cᵂu★ * u★³ - CᵂwΔ * wΔ³

where `u★` is the friction velocity and `wΔ` is the convective velocity scale.

This can be used alone at solid surfaces (ocean bottom, atmosphere bottom)
or wrapped in `TKEDissipationBoundaryCondition` for ocean surfaces.
"""
Base.@kwdef struct SurfaceTKEBoundaryCondition{FT}
    Cᵂu★ :: FT = 0.0
    CᵂwΔ :: FT = 0.0
end

"""
    WaveBreakingDissipationBoundaryCondition{FT}

Parameters for dissipation rate boundary condition at an ocean surface,
representing turbulence from breaking waves using the Charnock relation.
"""
Base.@kwdef struct WaveBreakingDissipationBoundaryCondition{FT}
    Cᵂα  :: FT = 0.11 # Charnock parameter
    gravitational_acceleration :: FT = 9.8065
    minimum_roughness_length :: FT = 1e-4
end

"""
    TKEDissipationBoundaryCondition{TKE, EPS}

A wrapper containing both TKE and dissipation boundary conditions.
Used at ocean surfaces where both e and ϵ require special treatment.

# Fields
- `tke`: A `SurfaceTKEBoundaryCondition` for TKE flux
- `dissipation`: A `WaveBreakingDissipationBoundaryCondition` for dissipation flux
"""
struct TKEDissipationBoundaryCondition{TKE, EPS}
    tke :: TKE
    dissipation :: EPS
end

"""
    TKEDissipationBoundaryCondition(FT = Float64)

Construct a `TKEDissipationBoundaryCondition` with default parameters.
This is the default for ocean surfaces.
"""
TKEDissipationBoundaryCondition(FT::DataType = Float64) =
    TKEDissipationBoundaryCondition(SurfaceTKEBoundaryCondition{FT}(),
                                    WaveBreakingDissipationBoundaryCondition{FT}())

# For backwards compatibility: alias the old name
const TKEOceanSurfaceBoundaryCondition = TKEDissipationBoundaryCondition

#####
##### Time-stepping TKE and dissipation equations
#####

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

    closure_fields = model.closure_fields
    κe = closure_fields.κe
    κϵ = closure_fields.κϵ
    Le = closure_fields.Le
    Lϵ = closure_fields.Lϵ
    previous_velocities = closure_fields.previous_velocities
    e_index = findfirst(k -> k == :e, keys(model.tracers))
    ϵ_index = findfirst(k -> k == :ϵ, keys(model.tracers))
    implicit_solver = model.timestepper.implicit_solver

    FT = eltype(model.tracers.e)
    Δt = convert(FT, model.clock.last_Δt)
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

        tracers = buoyancy_tracers(model)
        buoyancy = buoyancy_force(model)

        launch!(arch, grid, :xyz,
                compute_tke_dissipation_diffusivities!,
                κe, κϵ,
                grid, closure,
                model.velocities, tracers, buoyancy)

        # Compute the linear implicit component of the RHS (diffusivities, L)
        # and step forward
        launch!(arch, grid, :xyz,
                substep_tke_dissipation!,
                Le, Lϵ,
                grid, closure,
                model.velocities, previous_velocities, # try this soon: model.velocities, model.velocities,
                tracers, buoyancy, closure_fields,
                Δτ, χ, Gⁿe, G⁻e, Gⁿϵ, G⁻ϵ)

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

# Compute TKE and dissipation diffusivities
@kernel function compute_tke_dissipation_diffusivities!(κe, κϵ, grid, closure,
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
                                          tracers, buoyancy, diffusivities,
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
    wb = explicit_buoyancy_flux(i, j, k, grid, closure_ij, next_velocities, tracers, buoyancy, diffusivities)

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
    κu = diffusivities.κu
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
    Δτ = convert(FT, Δτ)
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
##### TKE flux computation (shared by all BC types)
#####

"""
    compute_tke_flux(i, j, grid, clock, fields, bc::SurfaceTKEBoundaryCondition,
                     buoyancy, top_tracer_bcs, top_velocity_bcs)

Compute TKE surface flux from friction velocity and convective velocity scale.
"""
@inline function compute_tke_flux(i, j, grid, clock, fields, bc::SurfaceTKEBoundaryCondition,
                                  buoyancy, top_tracer_bcs, top_velocity_bcs)
    wΔ³ = top_convective_turbulent_velocity_cubed(i, j, grid, clock, fields, buoyancy, top_tracer_bcs)
    u★ = friction_velocity(i, j, grid, clock, fields, top_velocity_bcs)
    Cᵂu★ = bc.Cᵂu★
    CᵂwΔ = bc.CᵂwΔ
    return - Cᵂu★ * u★^3 - CᵂwΔ * wΔ³
end

"""
    compute_dissipation_flux(i, j, k, grid, clock, fields, bc::WaveBreakingDissipationBoundaryCondition,
                             closure, buoyancy, top_velocity_bcs)

Compute dissipation rate surface flux from wave breaking (Charnock relation).
"""
@inline function compute_dissipation_flux(i, j, k, grid, clock, fields,
                                          bc::WaveBreakingDissipationBoundaryCondition,
                                          closure, buoyancy, top_velocity_bcs)
    𝕊u₀ = closure.stability_functions.𝕊u₀
    σϵ = closure.stability_functions.Cσϵ

    u★ = friction_velocity(i, j, grid, clock, fields, top_velocity_bcs)
    α = bc.Cᵂα
    g = bc.gravitational_acceleration
    ℓ_charnock = α * u★^2 / g

    ℓmin = bc.minimum_roughness_length
    ℓᵣ = max(ℓmin, ℓ_charnock)

    e★ = turbulent_kinetic_energyᶜᶜᶜ(i, j, k, grid, closure, fields)
    z = znode(i, j, k, grid, c, c, c)
    d = - z

    return - 𝕊u₀^4 / σϵ * e★^2 / (d + ℓᵣ)
end

#####
##### TKE boundary condition flux functions (dispatched by closure BC type)
#####

# Extract TKE BC from different boundary condition types
@inline get_tke_bc(bc::SurfaceTKEBoundaryCondition) = bc
@inline get_tke_bc(bc::TKEDissipationBoundaryCondition) = bc.tke

# Extract dissipation BC (only exists in TKEDissipationBoundaryCondition)
@inline get_dissipation_bc(bc::TKEDissipationBoundaryCondition) = bc.dissipation
@inline get_dissipation_bc(::SurfaceTKEBoundaryCondition) = nothing

#####
##### Top boundary condition flux functions
#####

@inline function top_tke_flux(i, j, grid, clock, fields, parameters, closure::FlavorOfTD, buoyancy)
    closure = getclosure(i, j, closure)
    top_bc = closure.top_boundary_condition
    tke_bc = get_tke_bc(top_bc)

    top_tracer_bcs = parameters.top_tracer_boundary_conditions
    top_velocity_bcs = parameters.top_velocity_boundary_conditions

    return compute_tke_flux(i, j, grid, clock, fields, tke_bc, buoyancy, top_tracer_bcs, top_velocity_bcs)
end

@inline function top_dissipation_flux(i, j, grid, clock, fields, parameters, closure::FlavorOfTD, buoyancy)
    closure = getclosure(i, j, closure)
    top_bc = closure.top_boundary_condition
    dissipation_bc = get_dissipation_bc(top_bc)

    top_velocity_bcs = parameters.top_velocity_boundary_conditions
    k = grid.Nz

    return _top_dissipation_flux(i, j, k, grid, clock, fields, dissipation_bc, closure, buoyancy, top_velocity_bcs)
end

# Dissipation flux with WaveBreakingDissipationBoundaryCondition
@inline function _top_dissipation_flux(i, j, k, grid, clock, fields,
                                       bc::WaveBreakingDissipationBoundaryCondition,
                                       closure, buoyancy, top_velocity_bcs)
    return compute_dissipation_flux(i, j, k, grid, clock, fields, bc, closure, buoyancy, top_velocity_bcs)
end

# No dissipation flux for SurfaceTKEBoundaryCondition alone
@inline _top_dissipation_flux(i, j, k, grid, clock, fields, ::Nothing, closure, buoyancy, top_velocity_bcs) = zero(grid)

#####
##### Bottom boundary condition flux functions
#####

@inline function bottom_tke_flux(i, j, grid, clock, fields, parameters, closure::FlavorOfTD, buoyancy)
    closure = getclosure(i, j, closure)
    bottom_bc = closure.bottom_boundary_condition
    tke_bc = get_tke_bc(bottom_bc)

    bottom_tracer_bcs = parameters.bottom_tracer_boundary_conditions
    bottom_velocity_bcs = parameters.bottom_velocity_boundary_conditions

    return compute_tke_flux(i, j, grid, clock, fields, tke_bc, buoyancy, bottom_tracer_bcs, bottom_velocity_bcs)
end

# Fallback for nothing bottom BC
@inline get_tke_bc(::Nothing) = nothing
@inline function compute_tke_flux(i, j, grid, clock, fields, ::Nothing, buoyancy, tracer_bcs, velocity_bcs)
    return zero(grid)
end

#####
##### Utilities for model constructors
#####

add_tke_dissipation_top_boundary_conditions(closure, user_bcs, args...) = user_bcs
add_tke_dissipation_bottom_boundary_conditions(closure, user_bcs, args...) = user_bcs

""" Add TKE boundary conditions specific to `TKEDissipationVerticalDiffusivity`. """
function add_closure_specific_boundary_conditions(closure::FlavorOfTD, user_bcs, grid, tracer_names, buoyancy)
    user_bcs = add_tke_dissipation_top_boundary_conditions(closure, user_bcs, grid, tracer_names, buoyancy)
    user_bcs = add_tke_dissipation_bottom_boundary_conditions(closure, user_bcs, grid, tracer_names, buoyancy)
    return user_bcs
end

#####
##### Top boundary condition: TKEDissipationBoundaryCondition (both TKE and dissipation)
#####

const TDWithOceanSurfaceTopBC = FlavorOfTD{<:Any, <:TKEDissipationBoundaryCondition}

function add_tke_dissipation_top_boundary_conditions(closure::TDWithOceanSurfaceTopBC,
                                                     user_bcs, grid, tracer_names, buoyancy)

    top_tracer_bcs = top_tracer_boundary_conditions(grid, tracer_names, user_bcs)
    top_velocity_bcs = top_velocity_boundary_conditions(grid, user_bcs)
    parameters = TKETopBoundaryConditionParameters(top_tracer_bcs, top_velocity_bcs)

    top_tke_bc = FluxBoundaryCondition(top_tke_flux, discrete_form=true, parameters=parameters)
    top_dissipation_bc = FluxBoundaryCondition(top_dissipation_flux, discrete_form=true, parameters=parameters)

    tke_bcs = merge_tke_boundary_conditions(grid, user_bcs, :top, top_tke_bc)
    dissipation_bcs = merge_dissipation_boundary_conditions(grid, user_bcs, :top, top_dissipation_bc)

    return merge(user_bcs, (e=tke_bcs, ϵ=dissipation_bcs))
end

#####
##### Top boundary condition: SurfaceTKEBoundaryCondition only (TKE flux, no dissipation flux)
#####

const TDWithSurfaceTKETopBC = FlavorOfTD{<:Any, <:SurfaceTKEBoundaryCondition}

function add_tke_dissipation_top_boundary_conditions(closure::TDWithSurfaceTKETopBC,
                                                     user_bcs, grid, tracer_names, buoyancy)

    top_tracer_bcs = top_tracer_boundary_conditions(grid, tracer_names, user_bcs)
    top_velocity_bcs = top_velocity_boundary_conditions(grid, user_bcs)
    parameters = TKETopBoundaryConditionParameters(top_tracer_bcs, top_velocity_bcs)

    top_tke_bc = FluxBoundaryCondition(top_tke_flux, discrete_form=true, parameters=parameters)

    tke_bcs = merge_tke_boundary_conditions(grid, user_bcs, :top, top_tke_bc)

    return merge(user_bcs, (e=tke_bcs,))
end

#####
##### Bottom boundary condition: TKEDissipationBoundaryCondition (both TKE and dissipation)
#####

const TDWithOceanSurfaceBottomBC = FlavorOfTD{<:Any, <:Any, <:TKEDissipationBoundaryCondition}

function add_tke_dissipation_bottom_boundary_conditions(closure::TDWithOceanSurfaceBottomBC,
                                                        user_bcs, grid, tracer_names, buoyancy)

    bottom_tracer_bcs = bottom_tracer_boundary_conditions(grid, tracer_names, user_bcs)
    bottom_velocity_bcs = bottom_velocity_boundary_conditions(grid, user_bcs)
    parameters = TKEBottomBoundaryConditionParameters(bottom_tracer_bcs, bottom_velocity_bcs)

    bottom_tke_bc = FluxBoundaryCondition(bottom_tke_flux, discrete_form=true, parameters=parameters)
    # Note: dissipation BC at bottom is generally not used (wave breaking is a surface phenomenon)
    # but we include it for completeness if someone wants it

    tke_bcs = merge_tke_boundary_conditions(grid, user_bcs, :bottom, bottom_tke_bc)

    return merge(user_bcs, (e=tke_bcs,))
end

#####
##### Bottom boundary condition: SurfaceTKEBoundaryCondition only
#####

const TDWithSurfaceTKEBottomBC = FlavorOfTD{<:Any, <:Any, <:SurfaceTKEBoundaryCondition}

function add_tke_dissipation_bottom_boundary_conditions(closure::TDWithSurfaceTKEBottomBC,
                                                        user_bcs, grid, tracer_names, buoyancy)

    bottom_tracer_bcs = bottom_tracer_boundary_conditions(grid, tracer_names, user_bcs)
    bottom_velocity_bcs = bottom_velocity_boundary_conditions(grid, user_bcs)
    parameters = TKEBottomBoundaryConditionParameters(bottom_tracer_bcs, bottom_velocity_bcs)

    bottom_tke_bc = FluxBoundaryCondition(bottom_tke_flux, discrete_form=true, parameters=parameters)

    tke_bcs = merge_tke_boundary_conditions(grid, user_bcs, :bottom, bottom_tke_bc)

    return merge(user_bcs, (e=tke_bcs,))
end

#####
##### Helper functions to merge boundary conditions
#####

function merge_tke_boundary_conditions(grid, user_bcs, location::Symbol, new_bc)
    if :e ∈ keys(user_bcs)
        e_bcs = user_bcs[:e]
        if location == :top
            return FieldBoundaryConditions(grid, (Center(), Center(), Center()),
                                           top = new_bc,
                                           bottom = e_bcs.bottom,
                                           north = e_bcs.north,
                                           south = e_bcs.south,
                                           east = e_bcs.east,
                                           west = e_bcs.west)
        else # :bottom
            return FieldBoundaryConditions(grid, (Center(), Center(), Center()),
                                           top = e_bcs.top,
                                           bottom = new_bc,
                                           north = e_bcs.north,
                                           south = e_bcs.south,
                                           east = e_bcs.east,
                                           west = e_bcs.west)
        end
    else
        if location == :top
            return FieldBoundaryConditions(grid, (Center(), Center(), Center()), top=new_bc)
        else
            return FieldBoundaryConditions(grid, (Center(), Center(), Center()), bottom=new_bc)
        end
    end
end

function merge_dissipation_boundary_conditions(grid, user_bcs, location::Symbol, new_bc)
    if :ϵ ∈ keys(user_bcs)
        ϵ_bcs = user_bcs[:ϵ]
        if location == :top
            return FieldBoundaryConditions(grid, (Center(), Center(), Center()),
                                           top = new_bc,
                                           bottom = ϵ_bcs.bottom,
                                           north = ϵ_bcs.north,
                                           south = ϵ_bcs.south,
                                           east = ϵ_bcs.east,
                                           west = ϵ_bcs.west)
        else # :bottom
            return FieldBoundaryConditions(grid, (Center(), Center(), Center()),
                                           top = ϵ_bcs.top,
                                           bottom = new_bc,
                                           north = ϵ_bcs.north,
                                           south = ϵ_bcs.south,
                                           east = ϵ_bcs.east,
                                           west = ϵ_bcs.west)
        end
    else
        if location == :top
            return FieldBoundaryConditions(grid, (Center(), Center(), Center()), top=new_bc)
        else
            return FieldBoundaryConditions(grid, (Center(), Center(), Center()), bottom=new_bc)
        end
    end
end

