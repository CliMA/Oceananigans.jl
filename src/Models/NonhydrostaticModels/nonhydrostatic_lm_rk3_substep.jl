using Oceananigans.TimeSteppers: _rk3_substep_field!, stage_Δt
using Oceananigans.Operators: divᶜᶜᶜ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ
using Oceananigans.TurbulenceClosures: ScalarDiffusivity
import Oceananigans.TimeSteppers: lm_rk3_substep!, seed_lm_pressures!

#####
##### Le & Moin (1991) viscous-divergence-correction term
#####
##### When intermediate uᵏ⁻¹ is not divergence-free, the explicit Laplacian carries
##### a spurious gradient component: ν∇²u = ν∇(∇·u) − ν∇×(∇×u). Subtracting
##### ν∇(∇·u) from the velocity tendency leaves only the solenoidal part, matching
##### what vanilla RK3 sees (where projection at every stage zeros ∇·u). The sign
##### below (Gu -=) is verified empirically against the Taylor–Green convergence
##### test in validation/convergence_tests/run_taylor_green_temporal.jl.

@inline lm_rk3_kinematic_viscosity(::Any) = nothing
@inline lm_rk3_kinematic_viscosity(::Nothing) = nothing
@inline lm_rk3_kinematic_viscosity(c::ScalarDiffusivity) = c.ν isa Number ? c.ν : nothing

@kernel function _compute_divergence!(δ, grid, u, v, w)
    i, j, k = @index(Global, NTuple)
    @inbounds δ[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

@kernel function _add_divergence_correction!(Gu, Gv, Gw, grid, δ, ν)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] -= ν * ∂xᶠᶜᶜ(i, j, k, grid, δ)
    @inbounds Gv[i, j, k] -= ν * ∂yᶜᶠᶜ(i, j, k, grid, δ)
    @inbounds Gw[i, j, k] -= ν * ∂zᶜᶜᶠ(i, j, k, grid, δ)
end

function apply_divergence_correction!(model::NonhydrostaticModel)
    ν = lm_rk3_kinematic_viscosity(model.closure)
    isnothing(ν) && return nothing
    grid = model.grid
    arch = architecture(grid)
    δ    = model.timestepper.divergence_buffer
    u, v, w = model.velocities
    Gu, Gv, Gw = model.timestepper.Gⁿ.u, model.timestepper.Gⁿ.v, model.timestepper.Gⁿ.w
    launch!(arch, grid, :xyz, _compute_divergence!, δ, grid, u, v, w)
    fill_halo_regions!(δ)
    launch!(arch, grid, :xyz, _add_divergence_correction!, Gu, Gv, Gw, grid, δ, ν)
    return nothing
end

#####
##### FPJ-2 pressure extrapolation (Capuano et al. 2016 / De Michele et al. 2020)
#####
##### Linear interpolation of φ assuming φⁿ and φⁿ⁻¹ are second-order approximations
##### of the true pressure at tⁿ − Δtⁿ⁻¹/2 and tⁿ⁻¹ − Δtⁿ⁻¹/2 (midpoint trick),
##### evaluated at tⁿ + cᵢ·Δtⁿ/2 to give the substage pressure predictor:
#####
#####   φᵢ ≈ φⁿ + (Δtⁿ/(2·Δtⁿ⁻¹))·(1 + cᵢ)·(φⁿ − φⁿ⁻¹)
#####
##### For constant Δt this reduces to φⁿ + ((1+cᵢ)/2)·(φⁿ − φⁿ⁻¹).

@kernel function _build_scaled_fpj2!(pNHS, pⁿ, pⁿ⁻¹, Δτ, scale)
    i, j, k = @index(Global, NTuple)
    @inbounds pNHS[i, j, k] = Δτ * pⁿ[i, j, k] + scale * (pⁿ[i, j, k] - pⁿ⁻¹[i, j, k])
end

# Substage time fractions cᵢ for Wray RK3 (where uᵢ corresponds to time tⁿ + cᵢ·Δt):
#   c₁ = γ¹ = 8/15
#   c₂ = γ¹ + γ² + ζ² = 2/3
#   c₃ = 1
@inline _fpj2_substage_c(ts, ::Val{1}) = ts.γ¹
@inline _fpj2_substage_c(ts, ::Val{2}) = ts.γ¹ + ts.γ² + ts.ζ²
@inline _fpj2_substage_c(ts, ::Val{3}) = one(ts.γ¹)

function apply_fpj2_pressure_correction!(model::NonhydrostaticModel, Δτ, stage::Val, Δtⁿ)
    ts   = model.timestepper
    pⁿ   = ts.pⁿ
    pⁿ⁻¹ = ts.pⁿ⁻¹
    pNHS = model.pressures.pNHS
    grid = model.grid
    arch = architecture(grid)

    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities, model.clock, fields(model))

    FT  = eltype(pNHS)
    cᵢ  = _fpj2_substage_c(ts, stage)
    Δtⁿ⁻¹ = ts.Δt⁻¹[]

    # scale = Δτ × (Δtⁿ / (2·Δtⁿ⁻¹)) × (1 + cᵢ)
    scale = convert(FT, Δτ * Δtⁿ * (1 + cᵢ) / (2 * Δtⁿ⁻¹))
    Δτ_FT = convert(FT, Δτ)

    launch!(arch, grid, :xyz, _build_scaled_fpj2!, pNHS, pⁿ, pⁿ⁻¹, Δτ_FT, scale)
    fill_halo_regions!(pNHS)
    make_pressure_correction!(model, Δτ)
    return nothing
end

function seed_lm_pressures!(model::NonhydrostaticModel)
    parent(model.timestepper.pⁿ)   .= parent(model.pressures.pNHS)
    parent(model.timestepper.pⁿ⁻¹) .= parent(model.pressures.pNHS)
    return nothing
end

function lm_rk3_substep!(model::NonhydrostaticModel, Δt, γⁿ, ζⁿ, callbacks,
                         stage::Union{Val{1}, Val{2}})
    Δτ   = stage_Δt(Δt, γⁿ, ζⁿ)
    grid = model.grid

    compute_flux_bc_tendencies!(model)
    apply_divergence_correction!(model)
    model_fields = prognostic_fields(model)

    for (i, name) in enumerate(keys(model_fields))
        field = model_fields[name]
        exclude_periphery = i < 4 # We assume that the first 3 fields are velocity / momentum variables
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[name], model.timestepper.G⁻[name])
        launch!(architecture(grid), grid, :xyz, _rk3_substep_field!, kernel_args...; exclude_periphery)

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       Val(i-3),
                       model.clock,
                       fields(model),
                       Δτ)
    end

    apply_fpj2_pressure_correction!(model, Δτ, stage, Δt)

    return nothing
end

function lm_rk3_substep!(model::NonhydrostaticModel, Δt, γⁿ, ζⁿ, callbacks, stage::Val{3})
    Δτ   = stage_Δt(Δt, γⁿ, ζⁿ)
    grid = model.grid

    compute_flux_bc_tendencies!(model)
    apply_divergence_correction!(model)
    model_fields = prognostic_fields(model)

    for (i, name) in enumerate(keys(model_fields))
        field = model_fields[name]
        exclude_periphery = i < 4
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[name], model.timestepper.G⁻[name])
        launch!(architecture(grid), grid, :xyz, _rk3_substep_field!, kernel_args...; exclude_periphery)

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       Val(i-3),
                       model.clock,
                       fields(model),
                       Δτ)
    end

    # Pressure-increment formulation. Apply the FPJ-2 predictor φ³_extrap, then
    # solve a single Poisson for the increment δp. The total correction at this
    # substage is u -= Δτ ∇(φ³_extrap + δp); we record φⁿ⁺¹ = φ³_extrap + (Δτ/Δt) δp
    # for the next step's FPJ-2 extrapolation.
    apply_fpj2_pressure_correction!(model, Δτ, stage, Δt)

    # Rotate stored pressures BEFORE pⁿ is overwritten with the φ³_extrap snapshot.
    parent(model.timestepper.pⁿ⁻¹) .= parent(model.timestepper.pⁿ)
    # Stash φ³_extrap (currently in pNHS) into pⁿ as a temp; combined with δp below
    # this becomes φⁿ⁺¹.
    parent(model.timestepper.pⁿ) .= parent(model.pressures.pNHS)

    compute_pressure_correction!(model, Δτ)
    make_pressure_correction!(model, Δτ)

    γ = convert(eltype(model.pressures.pNHS), Δτ / Δt)
    parent(model.pressures.pNHS) .= parent(model.timestepper.pⁿ) .+ γ .* parent(model.pressures.pNHS)
    fill_halo_regions!(model.pressures.pNHS)

    # Finalize: pⁿ ← φⁿ⁺¹, store this step's Δt as Δtⁿ⁻¹ for the next step.
    parent(model.timestepper.pⁿ) .= parent(model.pressures.pNHS)
    model.timestepper.Δt⁻¹[] = convert(eltype(model.timestepper.Δt⁻¹), Δt)

    return nothing
end
