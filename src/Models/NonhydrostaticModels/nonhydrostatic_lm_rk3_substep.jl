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

# FPJ-α/β family parameters (single source of truth, used by both the substage
# predictor kernel and the stage-3 storage formula). Set:
#   FPJ_α = 0,   FPJ_β = 0    → FPJ-0
#   FPJ_α = 1,   FPJ_β = 0    → FPJ-1
#   FPJ_α = 1//2, FPJ_β = 1//2 → FPJ-2
const FPJ_α = 1//2
const FPJ_β = 1//2

@kernel function _build_scaled_fpj2!(pNHS, pⁿ, pⁿ⁻¹, Δτ, cᵢ, cᵢ₋₁)
    i, j, k = @index(Global, NTuple)
    @inbounds μp = (pⁿ[i, j, k] + pⁿ⁻¹[i, j, k]) / 2
    @inbounds δp = pⁿ[i, j, k] - pⁿ⁻¹[i, j, k]
    α = FPJ_α
    β = FPJ_β
    # Low-storage scale for Wray RK3 (De Michele 2020 Eq. 37): cᵢ + cᵢ₋₁.
    scale = (1 + 2β) / 2 + α * (cᵢ + cᵢ₋₁)
    @inbounds pNHS[i, j, k] = Δτ * (μp + scale * δp)
end

# Substage time fractions cᵢ for Wray RK3 (where uᵢ corresponds to time tⁿ + cᵢ·Δt):
#   c₁ = γ¹ = 8/15
#   c₂ = γ¹ + γ² + ζ² = 2/3
#   c₃ = 1
@inline _fpj2_substage_c(ts, ::Val{1}) = ts.γ¹
@inline _fpj2_substage_c(ts, ::Val{2}) = ts.γ¹ + ts.γ² + ts.ζ²
@inline _fpj2_substage_c(ts, ::Val{3}) = one(ts.γ¹)

# Substage cumulative time fractions cᵢ for Wray RK3 (uᵢ corresponds to tⁿ + cᵢ·Δt):
#   c₀ = 0, c₁ = γ¹ = 8/15, c₂ = γ¹+γ²+ζ² = 2/3, c₃ = 1.
# Low-storage adaptation (De Michele 2020 Eq. 37) requires the predictor scale
# to be (cᵢ²−cᵢ₋₁²)/(cᵢ−cᵢ₋₁) = cᵢ + cᵢ₋₁, not cᵢ — see eq. (37) discussion in
# Sec. 3.3 of the paper. For Wray:
#   substage 1: c₁ + c₀ = 8/15
#   substage 2: c₂ + c₁ = 18/15
#   substage 3: c₃ + c₂ = 5/3
@inline _fpj2_substage_c_pair(ts, ::Val{1}) = (ts.γ¹, zero(ts.γ¹))
@inline _fpj2_substage_c_pair(ts, ::Val{2}) = (ts.γ¹ + ts.γ² + ts.ζ², ts.γ¹)
@inline _fpj2_substage_c_pair(ts, ::Val{3}) = (one(ts.γ¹), ts.γ¹ + ts.γ² + ts.ζ²)

# Σᵢ_{i=1,2} (Δτᵢ/Δt)·(cᵢ+cᵢ₋₁) — appears in the stage-3 storage formula as the
# weight on δφ. For Wray this is (8/15)² + (2/15)·(2/3+8/15) = 4/9.
@inline function _fpj2_substage_low_storage_scale_weighted_sum(ts)
    Δτ1 = ts.γ¹
    Δτ2 = ts.γ² + ts.ζ²
    c0  = zero(ts.γ¹)
    c1  = ts.γ¹
    c2  = ts.γ¹ + ts.γ² + ts.ζ²
    return Δτ1 * (c1 + c0) + Δτ2 * (c2 + c1)
end

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
    # cᵢ  = _fpj2_substage_c(ts, stage)
    cᵢ, cᵢ₋₁ = _fpj2_substage_c_pair(ts, stage)

    Δtⁿ⁻¹ = ts.Δt⁻¹[]

    # scale = convert(FT, cᵢ)
    Δτ_FT = convert(FT, Δτ)

    # launch!(arch, grid, :xyz, _build_scaled_fpj2!, pNHS, pⁿ, pⁿ⁻¹, Δτ_FT, scale)
    launch!(arch, grid, :xyz, _build_scaled_fpj2!, pNHS, pⁿ, pⁿ⁻¹, Δτ_FT, convert(FT, cᵢ), convert(FT, cᵢ₋₁))
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
    # apply_divergence_correction!(model)
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
    # apply_divergence_correction!(model)
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

    # Single full Poisson at substage 3 (Capuano 2016 Eq. 7 / De Michele 2020
    # Eqs. 19-21). Substages 1 and 2 used the FPJ predictor; substage 3 has
    # no predictor, just a full Poisson for δp_full.
    #
    # Stored φⁿ⁺¹ must satisfy Δt·∇φⁿ⁺¹ = Σᵢ Δτᵢ·∇φ_used_i so that the
    # whole step is equivalent to a single end-of-step projection with φⁿ⁺¹.
    # With substage-i predictor φ_extrap_i = μp + Aᵢ·δφ (low-storage form),
    # this gives
    #     φⁿ⁺¹ = B₀·μp + B₁·δφ + γ·δp_full
    # with γ = Δτ³/Δt = 1/3, B₀ = (Δτ¹+Δτ²)/Δt = 2/3, and
    #     B₁ = ((1+2β)/2)·B₀ + α·(Σᵢ_{i=1,2} Δτᵢ·sᵢ)/Δt
    # For Wray with low-storage sᵢ = cᵢ+cᵢ₋₁ the i=1,2 weighted sum
    # Σᵢ Δτᵢ·sᵢ/Δt = (8/15)² + (2/15)·(2/3+8/15) = 4/9.
    # Storing δp_full directly (B₀=B₁=0, γ=1) makes perturbations grow as
    # 2× per step on wall-bounded stiff cases; the correct formula above
    # kills perturbations exactly in one step.
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities, model.clock, fields(model))

    compute_pressure_correction!(model, Δτ)
    make_pressure_correction!(model, Δτ)
    fill_halo_regions!(model.pressures.pNHS)
    # pNHS now holds δp_full; pⁿ holds pⁿ_old; pⁿ⁻¹ holds pⁿ⁻¹_old.

    FT  = eltype(model.pressures.pNHS)
    ts  = model.timestepper
    α   = convert(FT, FPJ_α)
    β   = convert(FT, FPJ_β)
    γFT = convert(FT, Δτ / Δt)
    B₀  = one(FT) - γFT
    sum_Δτs = convert(FT, _fpj2_substage_low_storage_scale_weighted_sum(ts))
    B₁  = ((one(FT) + 2β) / 2) * B₀ + α * sum_Δτs

    pⁿ_arr   = parent(ts.pⁿ)
    pⁿ⁻¹_arr = parent(ts.pⁿ⁻¹)
    pNHS_arr = parent(model.pressures.pNHS)
    # Build pⁿ⁺¹ = B₀·μp + B₁·δφ + γ·δp_full into pNHS.
    pNHS_arr .= B₀ .* (pⁿ_arr .+ pⁿ⁻¹_arr) ./ 2 .+
                B₁ .* (pⁿ_arr .- pⁿ⁻¹_arr) .+
                γFT .* pNHS_arr
    # Rotate: pⁿ⁻¹ ← pⁿ_old, pⁿ ← pⁿ⁺¹.
    pⁿ⁻¹_arr .= pⁿ_arr
    pⁿ_arr   .= pNHS_arr

    model.timestepper.Δt⁻¹[] = convert(eltype(model.timestepper.Δt⁻¹), Δt)

    return nothing
end
