using KernelAbstractions: @kernel, @index
using Oceananigans: AbstractModel
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, fill_halo_regions!, regularize_field_boundary_conditions
using Oceananigans.Fields: XFaceField, YFaceField
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, SplitRungeKuttaTimeStepper
using Oceananigans.Utils: launch!

"""
    DualGridScheme(grid; relaxation_time = 10 * 86400)

C-D grid Coriolis scheme of Adcroft, Hill and Marshall (1999), Mon. Wea. Rev. 127, 1928-1936.
The Coriolis force on the C-grid velocities uses D-grid velocities located at the same points,
`vᴰ` at `u` points and `uᴰ` at `v` points, so that `f × u` is evaluated without spatial averaging.
The D-grid velocities are the C-grid velocities averaged to the D-grid points plus a prognostic deviation,
`uᴰ = ⟨u⟩ + δu` and `vᴰ = ⟨v⟩ + δv`, which evolves with the difference between the co-located and the averaged
Coriolis forces and relaxes toward zero: `∂t δu = f v - ⟨f vᴰ⟩ - δu / τ` and `∂t δv = ⟨f uᴰ⟩ - f u - δv / τ`.
The deviation is stepped with the `QuasiAdamsBashforth2TimeStepper` or the `SplitRungeKuttaTimeStepper` of a
`HydrostaticFreeSurfaceModel`.

Keyword arguments
=================

- `relaxation_time`: time scale `τ` [s] over which the D-grid velocities relax toward the averaged C-grid velocities.
  Default: 10 days. `Inf` disables the relaxation.
"""
struct DualGridScheme{D, S, FT}
    velocity_deviations :: D
    tendencies :: S
    timestepper_cache :: S
    relaxation_rate :: FT
end

function DualGridScheme(grid; relaxation_time = 10 * 86400)
    vbcs = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :u)
    ubcs = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :v)
    velocity_deviations, tendencies, timestepper_cache = Tuple((u = YFaceField(grid; boundary_conditions=ubcs),
                                                                v = XFaceField(grid; boundary_conditions=vbcs)) for _ in 1:3)

    relaxation_rate = convert(eltype(grid), 1 / relaxation_time)

    return DualGridScheme(velocity_deviations, tendencies, timestepper_cache, relaxation_rate)
end

Base.summary(::DualGridScheme) = "DualGridScheme"

Adapt.adapt_structure(to, scheme::DualGridScheme) = DualGridScheme(Adapt.adapt(to, scheme.velocity_deviations), nothing, nothing, scheme.relaxation_rate)

const DGS = AbstractRotation{<:DualGridScheme}

Oceananigans.prognostic_state(coriolis::DGS) = Oceananigans.prognostic_state((; coriolis.scheme.velocity_deviations, coriolis.scheme.timestepper_cache))
Oceananigans.restore_prognostic_state!(coriolis::DGS, from::NamedTuple) = Oceananigans.restore_prognostic_state!((; coriolis.scheme.velocity_deviations, coriolis.scheme.timestepper_cache), from)

@inline uᴰᶜᶠᶜ(i, j, k, grid, u, δu) = @inbounds (ℑxyᶜᶠᵃ(i, j, k, grid, u) + δu[i, j, k]) * !peripheral_node(i, j, k, grid, center, face, center)
@inline vᴰᶠᶜᶜ(i, j, k, grid, v, δv) = @inbounds (ℑxyᶠᶜᵃ(i, j, k, grid, v) + δv[i, j, k]) * !peripheral_node(i, j, k, grid, face, center, center)

@inline x_f_cross_U(i, j, k, grid, coriolis::DGS, U) = - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * vᴰᶠᶜᶜ(i, j, k, grid, U.v, coriolis.scheme.velocity_deviations.v)
@inline y_f_cross_U(i, j, k, grid, coriolis::DGS, U) = + ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * uᴰᶜᶠᶜ(i, j, k, grid, U.u, coriolis.scheme.velocity_deviations.u)

#####
##### Velocity deviations δ = uᴰ - ⟨u⟩: tendencies computed with the model tendencies, stepped at every model stage
#####

compute_coriolis_prognostic_tendencies!(coriolis, model) = nothing
step_coriolis_prognostics!(coriolis, model, Δt) = nothing

@kernel function _compute_velocity_deviation_tendencies!(Gδu, Gδv, grid, coriolis, velocities)
    i, j, k = @index(Global, NTuple)
    δu, δv = coriolis.scheme.velocity_deviations
    r = coriolis.scheme.relaxation_rate
    u, v = velocities.u, velocities.v
    @inbounds begin
        Gδu[i, j, k] = ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * v[i, j, k] + ℑxyᶜᶠᵃ(i, j, k, grid, x_f_cross_U, coriolis, velocities) - r * δu[i, j, k]
        Gδv[i, j, k] = ℑxyᶠᶜᵃ(i, j, k, grid, y_f_cross_U, coriolis, velocities) - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * u[i, j, k] - r * δv[i, j, k]
    end
end

function compute_coriolis_prognostic_tendencies!(coriolis::DGS, model)
    Gδu, Gδv = coriolis.scheme.tendencies
    grid = model.grid
    launch!(architecture(grid), grid, :xyz, _compute_velocity_deviation_tendencies!, Gδu, Gδv, grid, coriolis, model.velocities)
    return nothing
end

@kernel function _rk_substep_velocity_deviations!(δu, δv, δu⁰, δv⁰, Gδu, Gδv, Δτ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        δu[i, j, k] = δu⁰[i, j, k] + Δτ * Gδu[i, j, k]
        δv[i, j, k] = δv⁰[i, j, k] + Δτ * Gδv[i, j, k]
    end
end

function step_coriolis_prognostics!(coriolis::DGS, model::AbstractModel{<:SplitRungeKuttaTimeStepper}, Δτ)
    δ  = coriolis.scheme.velocity_deviations
    Gδ = coriolis.scheme.tendencies
    δ⁰ = coriolis.scheme.timestepper_cache
    grid = model.grid

    # At the first stage δ still holds the deviations at the beginning of the time step, from which every stage restarts
    if model.clock.stage == 1
        parent(δ⁰.u) .= parent(δ.u)
        parent(δ⁰.v) .= parent(δ.v)
    end

    launch!(architecture(grid), grid, :xyz, _rk_substep_velocity_deviations!, δ.u, δ.v, δ⁰.u, δ⁰.v, Gδ.u, Gδ.v, convert(eltype(grid), Δτ))
    fill_halo_regions!(δ)

    return nothing
end

@kernel function _ab2_step_velocity_deviations!(δu, δv, Gδu⁻, Gδv⁻, Gδu, Gδv, χ, Δt)
    i, j, k = @index(Global, NTuple)
    FT = eltype(δu)
    α = convert(FT, 3/2) + χ
    β = convert(FT, 1/2) + χ
    not_euler = χ != convert(FT, -0.5)
    @inbounds begin
        δu[i, j, k] += Δt * (α * Gδu[i, j, k] - β * Gδu⁻[i, j, k] * not_euler)
        δv[i, j, k] += Δt * (α * Gδv[i, j, k] - β * Gδv⁻[i, j, k] * not_euler)
        Gδu⁻[i, j, k] = Gδu[i, j, k]
        Gδv⁻[i, j, k] = Gδv[i, j, k]
    end
end

function step_coriolis_prognostics!(coriolis::DGS, model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt)
    δ   = coriolis.scheme.velocity_deviations
    Gδ  = coriolis.scheme.tendencies
    Gδ⁻ = coriolis.scheme.timestepper_cache
    grid = model.grid

    launch!(architecture(grid), grid, :xyz, _ab2_step_velocity_deviations!, δ.u, δ.v, Gδ⁻.u, Gδ⁻.v, Gδ.u, Gδ.v,
            model.timestepper.χ, convert(eltype(grid), Δt))
    fill_halo_regions!(δ)

    return nothing
end
