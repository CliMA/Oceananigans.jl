using Oceananigans.Utils: prettysummary

struct PerturbationAdvection{FT, D, TF, S} <: AbstractRadiationScheme{FT}
    inflow_timescale :: FT
    outflow_timescale :: FT
    gravity_wave_speed :: FT
    density :: D
    target_transport :: TF
    φᵇ :: S
end

"""
    PerturbationAdvection(FT = defaults.FloatType;
                          outflow_timescale = Inf,
                          inflow_timescale = 0,
                          gravity_wave_speed = 0,
                          density = nothing)

Create a `PerturbationAdvection` scheme, passed as the `scheme` keyword to an
`NormalFlowBoundaryCondition` (for boundary-normal velocities, `Face`-located) or to a
`ValueBoundaryCondition` (for scalars such as tracers, `Center`-located).
This scheme nudges the boundary value to the prescribed exterior value `val`,
using a time-scale `inflow_timescale` for inflow and `outflow_timescale` for outflow.

For cases where we assume that the internal flow is a small perturbation from
an external prescribed or coarser flow, we can split the velocity into background
and perturbation components.

We begin with the equation governing the fluid in the interior:

```math
∂ₜu + u ⋅ ∇u = -∇P + F
```

and note that on the boundary the pressure gradient is zero.
We can then assume that the flow composes of mean (`U⃗`) and perturbation (`u⃗'`) components,
and considering the `x`-component of velocity, we can rewrite the equation as

```math
\begin{align}
∂ₜu₁ &= -u₁ ∂₁u - u₂ ∂₂u₁ - u₃ ∂₃u₁ + F₁
      ≈ -U₁ ∂₁u₁' - U₂ ∂₂u₁' - U₃ ∂₃u₁' + F
\end{align}
```

Simplify by assuming that `U⃗ = U x̂`, and then take a numerical step to find `u₁`.
While derived for velocity, the resulting scheme generalizes to any prognostic field
`ψ` with prescribed exterior value `ψ̄`, advected by the boundary-normal velocity `U`
(for a boundary-normal velocity component, `U = ψ̄`). Denoting the boundary value
as `ψᴮ` and the adjacent interior value as `ψᴬ`, and noting that the
perturbation is `ψ' = ψ - ψ̄`, we take a backwards Euler step
on a right boundary:

```math
(ψ̄ⁿ⁺¹ - ψ̄ⁿ) / Δt + (ψ'ᴮⁿ⁺¹ - ψ'ᴮⁿ) / Δt = -Uⁿ⁺¹ (ψ'ᴮⁿ⁺¹ - ψ'ᴬⁿ⁺¹) / Δx + Fψ
```

This cannot be solved for general forcing, but if we assume the dominant forcing is
relaxation to the exterior value (i.e. `ψ' → 0`) then `Fψ = -ψ' / τ`,
and we can find `ψ'ᴮⁿ⁺¹`:

```math
ψ'ᴮⁿ⁺¹ = (ψᴮⁿ + Ũ ψ'ᴬⁿ⁺¹ - ψ̄ⁿ⁺¹) / (1 + Ũ + Δt / τ)
```

where `Ũ = U Δt / Δx`. Then `ψᴮⁿ⁺¹` is:

```math
ψᴮⁿ⁺¹ = (ψᴮⁿ + Ũ ψᴬⁿ⁺¹ + ψ̄ⁿ⁺¹ τ̃) / (1 + τ̃ + Ũ)
```

where `τ̃ = Δt / τ`.

The same operation can be repeated for left boundaries.

The relaxation timescale `τ` can be set to different values depending on whether
`U` points in or out of the domain (`inflow_timescale`/`outflow_timescale`). Since the
scheme is only valid when the flow is directed out of the domain the boundary condition
falls back to relaxation to the prescribed value. By default this happens instantly but
if the direction varies this may not be preferable. It is beneficial to relax the outflow
(i.e. non-zero `outflow_timescale`) to reduce the shock when the flow changes direction
to point into the domain.

The ideal value of the timescales probably depend on the grid spacing and details of the
boundary flow.

# Keyword Arguments

- `outflow_timescale`: relaxation timescale when flow exits the domain [s].
  Default: `Inf` (pure radiation, no relaxation).
- `inflow_timescale`: relaxation timescale when flow enters the domain [s].
  Default: `0` (instant relaxation to exterior value).
- `gravity_wave_speed`: additional phase speed added to the advective velocity [m/s].
  Useful for momentum fields where gravity waves propagate faster than the mean flow.
  Default: `0`.
- `density`: density field for converting density-weighted fields (ρψ) to intensive
  fields (ψ) before computing phase speeds and radiation. When provided, the scheme
  divides by `density` before radiation and multiplies back after. This is required
  for models with density-weighted prognostic variables (e.g., anelastic models with
  prognostic ρu, ρθ). Default: `nothing` (no conversion).
- `target_transport`: target net transport (m³/s) through this boundary, measured
  as the integral of the normal velocity in the positive coordinate direction.
  When set, the boundary velocity is corrected each pressure step to achieve exactly
  this flux before the global pool correction is applied to the remaining boundaries.
  Default: `nothing` (boundary participates in the global pool correction instead).
"""
function PerturbationAdvection(FT = defaults.FloatType;
                               outflow_timescale = Inf,
                               inflow_timescale = 0,
                               gravity_wave_speed = 0,
                               density = nothing,
                               target_transport = nothing)
    inflow_timescale = convert(FT, inflow_timescale)
    outflow_timescale = convert(FT, outflow_timescale)
    gravity_wave_speed = convert(FT, gravity_wave_speed)
    target_transport = convert_target_transport(FT, target_transport)
    return PerturbationAdvection(inflow_timescale, outflow_timescale, gravity_wave_speed, density, target_transport, nothing)
end

# Support 2-positional-arg constructor
PerturbationAdvection(inflow_timescale, outflow_timescale) =
    PerturbationAdvection(inflow_timescale, outflow_timescale, zero(inflow_timescale), nothing, nothing, nothing)

Adapt.adapt_structure(to, pe::PerturbationAdvection) =
    PerturbationAdvection(adapt(to, pe.inflow_timescale),
                          adapt(to, pe.outflow_timescale),
                          adapt(to, pe.gravity_wave_speed),
                          adapt(to, pe.density),
                          adapt(to, pe.target_transport),
                          adapt(to, pe.φᵇ))

radiation_storage(pe::PerturbationAdvection, buffer) =
    PerturbationAdvection(pe.inflow_timescale, pe.outflow_timescale, pe.gravity_wave_speed, pe.density, pe.target_transport, buffer())

Base.summary(::PerturbationAdvection{FT}) where FT = "PerturbationAdvection{$FT}"

function Base.show(io::IO, pe::PerturbationAdvection)
    print(io, summary(pe), '\n')
    print(io, "├── inflow_timescale: ", prettysummary(pe.inflow_timescale), '\n')
    print(io, "├── outflow_timescale: ", prettysummary(pe.outflow_timescale), '\n')
    print(io, "├── gravity_wave_speed: ", prettysummary(pe.gravity_wave_speed), '\n')
    print(io, "├── density: ", prettysummary(pe.density), '\n')
    print(io, "└── target_transport: ", prettysummary(pe.target_transport))
end

# PerturbationAdvection lives on `NormalFlow` for boundary-normal velocities (Face-located)
# and on `Value` for scalars such as tracers (Center-located).
const PANFBC = BoundaryCondition{<:NormalFlow{<:PerturbationAdvection}}
const PAVBC  = BoundaryCondition{<:Value{<:PerturbationAdvection}}
const PABC   = Union{PANFBC, PAVBC}

"""
$(TYPEDSIGNATURES)

Return `true` if `scheme` carries a prescribed `target_transport`, `false` otherwise.
"""
has_target_transport(scheme) = false
has_target_transport(scheme::PerturbationAdvection{<:Any, <:Any, <:Nothing}) = false
has_target_transport(scheme::PerturbationAdvection) = true  # any non-Nothing TF

"""
$(TYPEDSIGNATURES)

Return the prescribed target transport for `scheme` on `grid`.
For a `Number`, returns the stored value unchanged.
For a callable, calls it with `grid` to recompute the target at the current grid state.
"""
get_target_transport(scheme, grid) = get_target_transport(scheme)  # 1-arg fallback for external schemes

_eval_tt(x::Number, grid) = x
_eval_tt(f, grid) = f(grid)

# Helper to convert between density-weighted and intensive fields.
# When density is nothing, these are no-ops.
@inline to_intensive(::Nothing, ψ, k) = ψ
@inline to_intensive(ρ, ψ, k) = ψ / @inbounds ρ[1, 1, k]
@inline to_extensive(::Nothing, ψ, k) = ψ
@inline to_extensive(ρ, ψ, k) = @inbounds ρ[1, 1, k] * ψ

# Backward-Euler step from the base of the stage; the phase speed is the outward velocity plus `gravity_wave_speed`
@inline function step_exterior_state!(side, i, j, grid, ψ, bc::PABC, loc, clock, model_fields, Δt)
    iᵇ, jᵇ, kᵇ = boundary_point(side, loc, grid, 0, i, j)
    i₁, j₁, k₁ = boundary_point(side, loc, grid, 1, i, j)

    pa = bc.classification.scheme
    ρ = pa.density

    ψᵇ = to_intensive(ρ, @inbounds(pa.φᵇ[i, j]), kᵇ)
    ψ₁ = to_intensive(ρ, @inbounds(ψ[i₁, j₁, k₁]), kᵇ)
    ψ̄  = getbc(bc, i, j, grid, clock, model_fields)

    U = outward(side, advecting_velocity(bc, side, model_fields, grid, 0, i, j, ψ̄))
    c = U + pa.gravity_wave_speed
    Ũ = clamp(Δt * reciprocal_boundary_spacing(side, grid, i, j) * c, zero(c), one(c))

    τ = ifelse(U >= 0, pa.outflow_timescale, pa.inflow_timescale)
    τ̃ = Δt / τ

    ψⁿ⁺¹ = (ψᵇ + Ũ * ψ₁ + ψ̄ * τ̃) / (1 + τ̃ + Ũ)
    @inbounds ψ[iᵇ, jᵇ, kᵇ] = to_extensive(ρ, ifelse(τ == 0, ψ̄, ψⁿ⁺¹), kᵇ)

    return nothing
end
