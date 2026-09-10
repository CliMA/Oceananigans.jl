#####
##### ObliqueRadiation (Raymond & Kuo 1984) open boundary scheme
#####

"""
    ObliqueRadiation(; inflow_timescale = 1day, outflow_timescale = 360days, use_boundary_velocity = false)

Raymond & Kuo (1984) two-dimensional radiation condition with adaptive nudging
(Marchesiello et al. 2001) — the oblique generalization of [`NormalRadiation`](@ref).

`NormalRadiation` diagnoses a phase speed in the boundary-normal direction only, which is
optimal when a signal arrives perpendicular to the boundary and degrades as the angle of
incidence grows. `ObliqueRadiation` diagnoses both components of the phase-speed vector,

    ∂φ/∂t + cₙ ∂φ/∂n + cₜ ∂φ/∂t̂ = - (φ - φᵉˣᵗ) / τ

with

    cₙ = -(∂φ/∂t)(∂φ/∂n) / |∇φ|² ,   cₜ = -(∂φ/∂t)(∂φ/∂t̂) / |∇φ|² ,   |∇φ|² = (∂φ/∂n)² + (∂φ/∂t̂)²

where `n` is the boundary-normal and `t̂` the boundary-tangential (horizontal) direction. The
tangential derivative is taken upwind, following ROMS. `cₜ` is limited to a tangential Courant
number of one for stability.

When the tangential gradient vanishes, `|∇φ|² = (∂φ/∂n)²` and `cₙ` reduces exactly to the
`NormalRadiation` phase speed — the two schemes agree at normal incidence, and this is
verified in the test suite.

As in `NormalRadiation`, inflow versus outflow is decided from the boundary-normal velocity
rather than the sign of the diagnosed phase speed (a vanishing gradient makes the phase-speed
ratio blow up and flip sign as an extremum exits), `cₙ` and `cₜ` are set to zero on inflow, and
nudging is adaptive: `τ = τ_in` on inflow, `τ = τ_out` on outflow.

The nudging timescales default to `NormalRadiation`'s — `τ_in = 0` (snap to the exterior value
on inflow) and `τ_out = Inf` (pure radiation, no nudging on outflow) — so that swapping one
scheme for the other changes the phase-speed diagnosis and nothing else.

Marchesiello et al. (2001) recommend `τ_in = 1 day` and `τ_out = 360 days` for long-term
regional integrations, and those are worth trying. The right `τ_in` depends on how much the
exterior data is trusted: their values assume it is a coarse climatology you would rather nudge
toward than impose, whereas if the exterior data is accurate — a parent model in a nested
configuration, say — clamping with `τ_in = 0` is better, since any freedom given to the boundary
is freedom to be wrong. It is a knob to tune against the data in hand, not a constant.

Oblique radiation is applied on the four LATERAL boundaries, where the tangential direction is
horizontal. On the top and bottom boundaries the scheme falls back to the one-dimensional
`NormalRadiation` kernel — radiating a vertical boundary obliquely is not meaningful for a
hydrostatic ocean model, and ROMS likewise applies the two-dimensional form only laterally.

Storage is identical to `NormalRadiation`: the previous-timestep boundary and interior values
are held as two-dimensional slabs over the boundary face, so the tangential neighbours the
oblique term needs are already available and no extra allocation is required.

References
==========
* Raymond, W. H., & Kuo, H. L. (1984). "A radiation boundary condition for multi-dimensional
  flows." Quarterly Journal of the Royal Meteorological Society, 110(464), 535-551.
* Orlanski, I. (1976). "A simple boundary condition for unbounded hyperbolic flows."
  Journal of Computational Physics, 21(3), 251-269.
* Marchesiello, P., McWilliams, J. C., & Shchepetkin, A. (2001). "Open boundary conditions
  for long-term integration of regional oceanic models." Ocean Modelling, 3(1-2), 1-20.

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: ObliqueRadiation

ObliqueRadiation()

# output
ObliqueRadiation{Float64}
├── inflow_timescale: 0.0
├── outflow_timescale: Inf
└── use_boundary_velocity: false
```
"""
struct ObliqueRadiation{FT, S}
    outflow_timescale :: FT
    inflow_timescale  :: FT
    use_boundary_velocity :: Bool
    φᵇ  :: S  # anchor boundary value (2D array or nothing)
    φ₁  :: S  # anchor interior value (2D array or nothing)
    φ₁ˡ :: S  # latest interior value (2D array or nothing)
end

function ObliqueRadiation(FT = defaults.FloatType;
                          inflow_timescale = 0,
                          outflow_timescale = Inf,
                          use_boundary_velocity = false)

    outflow_timescale = convert(FT, outflow_timescale)
    inflow_timescale = convert(FT, inflow_timescale)
    return ObliqueRadiation(outflow_timescale, inflow_timescale, use_boundary_velocity,
                            nothing, nothing, nothing)
end

Adapt.adapt_structure(to, r::ObliqueRadiation) =
    ObliqueRadiation(adapt(to, r.outflow_timescale),
                     adapt(to, r.inflow_timescale),
                     r.use_boundary_velocity,
                     adapt(to, r.φᵇ),
                     adapt(to, r.φ₁),
                     adapt(to, r.φ₁ˡ))

Base.summary(::ObliqueRadiation{FT}) where FT = "ObliqueRadiation{$FT}"

function Base.show(io::IO, r::ObliqueRadiation)
    print(io, summary(r), '\n')
    print(io, "├── inflow_timescale: ",  prettysummary(r.inflow_timescale), '\n')
    print(io, "├── outflow_timescale: ", prettysummary(r.outflow_timescale), '\n')
    print(io, "└── use_boundary_velocity: ", r.use_boundary_velocity)
end

const ORVBC  = BoundaryCondition{<:Value{<:ObliqueRadiation}}
const ORNFBC = BoundaryCondition{<:NormalFlow{<:ObliqueRadiation}}
const ORBC   = Union{ORVBC, ORNFBC}

#####
##### Storage allocation during BC regularization — identical to NormalRadiation
#####

function materialize_radiation_storage(radiation::ObliqueRadiation, grid, loc, dim)
    FT = eltype(grid)
    Sx, Sy, Sz = size(grid, loc)
    arch = architecture(grid)

    tangential_size = dim == 1 ? (Sy, Sz) :
                      dim == 2 ? (Sx, Sz) :
                                 (Sx, Sy)

    φᵇ  = on_architecture(arch, zeros(FT, tangential_size...))
    φ₁  = on_architecture(arch, zeros(FT, tangential_size...))
    φ₁ˡ = on_architecture(arch, zeros(FT, tangential_size...))

    return ObliqueRadiation(radiation.outflow_timescale,
                            radiation.inflow_timescale,
                            radiation.use_boundary_velocity,
                            φᵇ, φ₁, φ₁ˡ)
end

function regularize_boundary_condition(bc::ORBC, grid, loc, dim, args...)
    regularized_condition = regularize_boundary_condition(bc.condition, grid, loc, dim, args...)
    radiation = bc.classification.scheme
    materialized_radiation = materialize_radiation_storage(radiation, grid, loc, dim)
    classification = rebuild_classification(bc.classification, materialized_radiation)
    return BoundaryCondition(classification, regularized_condition)
end

#####
##### The Raymond & Kuo (1984) kernel
#####

# Ported from ROMS `ROMS/Nonlinear/u3dbc_im.F` (western-edge block, the `RADIATION_2D`
# branch), which is the reference implementation of Marchesiello et al. (2001).
# ROMS is MIT/X licensed. Sign conventions differ: ROMS forms
#
#     dUdt = u₁ⁿ - u₁ⁿ⁺¹      (i.e. -∂φ/∂t)
#     dUdx = u₁ⁿ⁺¹ - u₂ⁿ⁺¹    (i.e.  ∂φ/∂ξ, inward)
#
# so ROMS's `Cx = dUdt*dUdx` equals `-∂t_φ * ∂ξ_φ` here, and likewise for `Ce`. ROMS keeps
# `cff = |∇φ|²` unnormalized and divides by `(cff + Cx)`; dividing numerator and denominator
# through by `cff` gives the normalized form below, which matches Oceananigans' existing
# `orlanski_radiation` structure exactly.
#
# Tangential differences are BACKWARD (`grad₋`) and FORWARD (`grad₊`) at the boundary and
# first-interior columns, evaluated at time n. `∂η_φ` selects the upwind one
# (ROMS u3dbc_im.F:135-140).
#
# Two limiters, both kept:
#   * `Cₙ` clamped to [0, 1] (Oceananigans convention), blended with the advective Courant
#     number `Cᵃ` — see the note in normal_radiation.jl on extrema exiting the boundary.
#   * `Cₜ` clamped to [-1, 1], which is ROMS's `Ce = min(cff, max(dUdt*dUde, -cff))`
#     after normalization. Essential for stability.
#
# Reduction check: when `∂η_φ = 0`, `D = ∂ξ_φ²` and `Cᶜ = -∂t_φ ∂ξ_φ / ∂ξ_φ² = -∂t_φ / ∂ξ_φ`,
# identical to the `NormalRadiation` phase speed, and `Cₜ = 0` removes the tangential terms.
# The two schemes therefore agree exactly at normal incidence.

@inline function raymond_kuo_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ,
                                       gradᵇ₋, gradᵇ₊, grad₁₋, grad₁₊,
                                       φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)

    ∂t_φ = φ₁ⁿ⁺¹ - φ₁ⁿ
    ∂ξ_φ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹

    # Upwind tangential gradient. ROMS tests `dUdt*(grad₁₋ + grad₁₊) > 0` with
    # `dUdt = -∂t_φ`, hence the flipped inequality here.
    ∂η_φ = ifelse(-∂t_φ * (grad₁₋ + grad₁₊) > 0, grad₁₋, grad₁₊)

    FT = typeof(∂ξ_φ)
    D  = max(∂ξ_φ^2 + ∂η_φ^2, eps(FT))

    Cᶜ = - ∂t_φ * ∂ξ_φ / D
    Cₙ = ifelse(outflow, max(zero(FT), min(one(FT), max(Cᶜ, Cᵃ))), zero(FT))
    Cₜ = ifelse(outflow, clamp(- ∂t_φ * ∂η_φ / D, -one(FT), one(FT)), zero(FT))

    τ  = ifelse(outflow, radiation.outflow_timescale, radiation.inflow_timescale)
    τ̃  = Δt / τ

    # Implicit Raymond-Kuo radiation + nudging
    φᵇⁿ⁺¹ = (φᵇⁿ + Cₙ * φ₁ⁿ⁺¹
             - max(Cₜ, zero(FT)) * gradᵇ₋
             - min(Cₜ, zero(FT)) * gradᵇ₊
             + τ̃ * φᵉˣᵗ) / (1 + Cₙ + τ̃)

    return ifelse(τ == 0, φᵉˣᵗ, φᵇⁿ⁺¹)
end

# Tangential neighbours of the stored 2D slabs, clamped at the ends of the boundary face.
# At the two corner cells there is no tangential neighbour on one side, so the corresponding
# difference is zero and the scheme degrades gracefully to the one-dimensional form there.
@inline function tangential_differences(φslab, t, k)
    T = size(φslab, 1)
    @inbounds begin
        φ₀ = φslab[t, k]
        φ₋ = φslab[max(t - 1, 1), k]
        φ₊ = φslab[min(t + 1, T), k]
    end
    return (φ₀ - φ₋, φ₊ - φ₀)   # (backward, forward)
end

#####
##### Halo filling — lateral boundaries
#####

@inline function oblique_radiate_east_halo!(iᵇ, j, k, grid, c, bc, Uₙ, loc, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme
    ℓx, ℓy, ℓz = loc

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, j, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[iᵇ-1, j, k]
        φ₂ⁿ⁺¹ = c[iᵇ-2, j, k]

        φᵇᵃ = ifelse(anchored, c[iᵇ, j, k], radiation.φᵇ[j, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[j, k], radiation.φ₁[j, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)

        gradᵇ₋, gradᵇ₊ = tangential_differences(radiation.φᵇ, j, k)
        grad₁₋, grad₁₊ = tangential_differences(radiation.φ₁, j, k)

        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δxᶠᶜᶜ(iᵇ, j, k, grid)
        outflow = Uᵃ >= 0

        φᵇⁿ⁺¹ = raymond_kuo_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ,
                                      gradᵇ₋, gradᵇ₊, grad₁₋, grad₁₊,
                                      φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
        closed = immersed_peripheral_node(grid.Nx, j, k, grid, Center(), ℓy, ℓz)
        c[iᵇ, j, k]         = ifelse(closed, zero(grid), φᵇⁿ⁺¹)
        radiation.φᵇ[j, k]  = φᵇⁿ
        radiation.φ₁[j, k]  = φ₁ⁿ
        radiation.φ₁ˡ[j, k] = φ₁ⁿ⁺¹
    end

    return nothing
end

@inline function oblique_radiate_west_halo!(iᵇ, j, k, grid, c, bc, Uₙ, loc, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme
    ℓx, ℓy, ℓz = loc

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, j, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[iᵇ+1, j, k]
        φ₂ⁿ⁺¹ = c[iᵇ+2, j, k]

        φᵇᵃ = ifelse(anchored, c[iᵇ, j, k], radiation.φᵇ[j, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[j, k], radiation.φ₁[j, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)

        gradᵇ₋, gradᵇ₊ = tangential_differences(radiation.φᵇ, j, k)
        grad₁₋, grad₁₊ = tangential_differences(radiation.φ₁, j, k)

        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δxᶠᶜᶜ(iᵇ + 1, j, k, grid)
        outflow = Uᵃ <= 0

        φᵇⁿ⁺¹ = raymond_kuo_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ,
                                      gradᵇ₋, gradᵇ₊, grad₁₋, grad₁₊,
                                      φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
        closed = immersed_peripheral_node(1, j, k, grid, Center(), ℓy, ℓz)
        c[iᵇ, j, k]         = ifelse(closed, zero(grid), φᵇⁿ⁺¹)
        radiation.φᵇ[j, k]  = φᵇⁿ
        radiation.φ₁[j, k]  = φ₁ⁿ
        radiation.φ₁ˡ[j, k] = φ₁ⁿ⁺¹
    end

    return nothing
end

@inline function oblique_radiate_north_halo!(jᵇ, i, k, grid, c, bc, Uₙ, loc, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme
    ℓx, ℓy, ℓz = loc

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, jᵇ-1, k]
        φ₂ⁿ⁺¹ = c[i, jᵇ-2, k]

        φᵇᵃ = ifelse(anchored, c[i, jᵇ, k], radiation.φᵇ[i, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[i, k], radiation.φ₁[i, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)

        gradᵇ₋, gradᵇ₊ = tangential_differences(radiation.φᵇ, i, k)
        grad₁₋, grad₁₊ = tangential_differences(radiation.φ₁, i, k)

        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δyᶜᶠᶜ(i, jᵇ, k, grid)
        outflow = Uᵃ >= 0

        φᵇⁿ⁺¹ = raymond_kuo_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ,
                                      gradᵇ₋, gradᵇ₊, grad₁₋, grad₁₊,
                                      φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
        closed = immersed_peripheral_node(i, grid.Ny, k, grid, ℓx, Center(), ℓz)
        c[i, jᵇ, k]         = ifelse(closed, zero(grid), φᵇⁿ⁺¹)
        radiation.φᵇ[i, k]  = φᵇⁿ
        radiation.φ₁[i, k]  = φ₁ⁿ
        radiation.φ₁ˡ[i, k] = φ₁ⁿ⁺¹
    end

    return nothing
end

@inline function oblique_radiate_south_halo!(jᵇ, i, k, grid, c, bc, Uₙ, loc, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme
    ℓx, ℓy, ℓz = loc

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i, jᵇ+1, k]
        φ₂ⁿ⁺¹ = c[i, jᵇ+2, k]

        φᵇᵃ = ifelse(anchored, c[i, jᵇ, k], radiation.φᵇ[i, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[i, k], radiation.φ₁[i, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)

        gradᵇ₋, gradᵇ₊ = tangential_differences(radiation.φᵇ, i, k)
        grad₁₋, grad₁₊ = tangential_differences(radiation.φ₁, i, k)

        Uᵃ  = advecting_velocity(Uₙ, φ₁ⁿ⁺¹)
        Cᵃ  = abs(Uᵃ) * Δt / Δyᶜᶠᶜ(i, jᵇ + 1, k, grid)
        outflow = Uᵃ <= 0

        φᵇⁿ⁺¹ = raymond_kuo_radiation(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ,
                                      gradᵇ₋, gradᵇ₊, grad₁₋, grad₁₊,
                                      φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
        closed = immersed_peripheral_node(i, 1, k, grid, ℓx, Center(), ℓz)
        c[i, jᵇ, k]         = ifelse(closed, zero(grid), φᵇⁿ⁺¹)
        radiation.φᵇ[i, k]  = φᵇⁿ
        radiation.φ₁[i, k]  = φ₁ⁿ
        radiation.φ₁ˡ[i, k] = φ₁ⁿ⁺¹
    end

    return nothing
end

#####
##### Dispatch
#####

# NormalFlow fields radiate with their own boundary value as advecting velocity (Uₙ = nothing).
@inline  _fill_east_halo!(j, k, grid, c, bc::ORNFBC, loc::FAA, clock, model_fields) =  oblique_radiate_east_halo!(grid.Nx+1, j, k, grid, c, bc, nothing, loc, clock, model_fields)
@inline  _fill_west_halo!(j, k, grid, c, bc::ORNFBC, loc::FAA, clock, model_fields) =  oblique_radiate_west_halo!(1,         j, k, grid, c, bc, nothing, loc, clock, model_fields)
@inline _fill_north_halo!(i, k, grid, c, bc::ORNFBC, loc::AFA, clock, model_fields) = oblique_radiate_north_halo!(grid.Ny+1, i, k, grid, c, bc, nothing, loc, clock, model_fields)
@inline _fill_south_halo!(i, k, grid, c, bc::ORNFBC, loc::AFA, clock, model_fields) = oblique_radiate_south_halo!(1,         i, k, grid, c, bc, nothing, loc, clock, model_fields)

# Vertical boundaries: no meaningful oblique direction, so use the 1-D kernel. `radiate_*_halo!`
# only touches `outflow_timescale` / `inflow_timescale` on the scheme, which `ObliqueRadiation`
# shares with `NormalRadiation`.
@inline    _fill_top_halo!(i, j, grid, c, bc::ORNFBC, loc::AAF, clock, model_fields) =    radiate_top_halo!(grid.Nz+1, i, j, grid, c, bc, nothing, loc, clock, model_fields)
@inline _fill_bottom_halo!(i, j, grid, c, bc::ORNFBC, loc::AAF, clock, model_fields) = radiate_bottom_halo!(1,         i, j, grid, c, bc, nothing, loc, clock, model_fields)

# Value fields (tracers, tangential velocities) are advected by the boundary-normal velocity.
@inline _fill_east_halo!(j, k, grid, c, bc::ORVBC, loc::CAA, clock, model_fields) =
    oblique_radiate_east_halo!(grid.Nx+1, j, k, grid, c, bc, @inbounds(model_fields.u[radiation_velocity_index(bc, grid.Nx+1, grid.Nx), j, k]), loc, clock, model_fields)
@inline _fill_west_halo!(j, k, grid, c, bc::ORVBC, loc::CAA, clock, model_fields) =
    oblique_radiate_west_halo!(0, j, k, grid, c, bc, @inbounds(model_fields.u[radiation_velocity_index(bc, 1, 2), j, k]), loc, clock, model_fields)
@inline _fill_north_halo!(i, k, grid, c, bc::ORVBC, loc::ACA, clock, model_fields) =
    oblique_radiate_north_halo!(grid.Ny+1, i, k, grid, c, bc, @inbounds(model_fields.v[i, radiation_velocity_index(bc, grid.Ny+1, grid.Ny), k]), loc, clock, model_fields)
@inline _fill_south_halo!(i, k, grid, c, bc::ORVBC, loc::ACA, clock, model_fields) =
    oblique_radiate_south_halo!(0, i, k, grid, c, bc, @inbounds(model_fields.v[i, radiation_velocity_index(bc, 1, 2), k]), loc, clock, model_fields)
@inline _fill_top_halo!(i, j, grid, c, bc::ORVBC, loc::AAC, clock, model_fields) =
    radiate_top_halo!(grid.Nz+1, i, j, grid, c, bc, @inbounds(model_fields.w[i, j, radiation_velocity_index(bc, grid.Nz+1, grid.Nz)]), loc, clock, model_fields)
@inline _fill_bottom_halo!(i, j, grid, c, bc::ORVBC, loc::AAC, clock, model_fields) =
    radiate_bottom_halo!(0, i, j, grid, c, bc, @inbounds(model_fields.w[i, j, radiation_velocity_index(bc, 1, 2)]), loc, clock, model_fields)
