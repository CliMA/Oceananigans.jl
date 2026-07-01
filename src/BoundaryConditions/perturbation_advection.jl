using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ
using Oceananigans: defaults
using Oceananigans.Utils: prettysummary

struct PerturbationAdvection{FT, D, TF}
    inflow_timescale :: FT
    outflow_timescale :: FT
    gravity_wave_speed :: FT
    density :: D
    target_transport :: TF
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
∂ₜu₁ = -u₁ ∂₁u - u₂ ∂₂u₁ - u₃ ∂₃u₁ + F₁
      ≈ -U₁ ∂₁u₁' - U₂ ∂₂u₁' - U₃ ∂₃u₁' + F
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
    tt = if isnothing(target_transport)
             nothing
         elseif target_transport isa Number
             convert(FT, target_transport)
         else
             target_transport  # callable of the grid; keep as-is
         end
    return PerturbationAdvection(inflow_timescale, outflow_timescale, gravity_wave_speed, density, tt)
end

# Support 2-positional-arg constructor
PerturbationAdvection(inflow_timescale, outflow_timescale) =
    PerturbationAdvection(inflow_timescale, outflow_timescale, zero(inflow_timescale), nothing, nothing)

Adapt.adapt_structure(to, pe::PerturbationAdvection) =
    PerturbationAdvection(adapt(to, pe.inflow_timescale),
                          adapt(to, pe.outflow_timescale),
                          adapt(to, pe.gravity_wave_speed),
                          adapt(to, pe.density),
                          adapt(to, pe.target_transport))

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
get_target_transport(scheme::PerturbationAdvection, grid) = _eval_tt(scheme.target_transport, grid)
get_target_transport(scheme::PerturbationAdvection) = scheme.target_transport

_eval_tt(x::Number, grid) = x
_eval_tt(f, grid) = f(grid)

# Helper to convert between density-weighted and intensive fields.
# When density is nothing, these are no-ops.
@inline to_intensive(::Nothing, ψ, k) = ψ
@inline to_intensive(ρ, ψ, k) = ψ / @inbounds ρ[1, 1, k]
@inline to_extensive(::Nothing, ψ, k) = ψ
@inline to_extensive(ρ, ψ, k) = @inbounds ρ[1, 1, k] * ψ

# Advecting velocity for the radiation phase speed: a boundary-normal velocity
# component advects itself (falls back to ψ̄), a tracer is advected by the flow.
@inline advecting_velocity(::Nothing, ψ̄) = ψ̄
@inline advecting_velocity(U, ψ̄) = U

@inline function step_right_open_boundary!(bc::PABC, l, m, boundary_indices, boundary_adjacent_indices,
                                           grid, ψ, U, clock, model_fields, ΔX, k)
    iᴮ, jᴮ, kᴮ = boundary_indices
    iᴬ, jᴬ, kᴬ = boundary_adjacent_indices
    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), zero(Δt), Δt)

    pa = bc.classification.scheme
    ρ = pa.density
    c★ = pa.gravity_wave_speed

    # Convert to intensive space (no-op when density is nothing)
    ψᴮ = to_intensive(ρ, @inbounds(ψ[iᴮ, jᴮ, kᴮ]), k)
    ψᴬ = to_intensive(ρ, @inbounds(ψ[iᴬ, jᴬ, kᴬ]), k)

    # Prescribed exterior value (in intensive units when density is provided)
    ψ̄ = getbc(bc, l, m, grid, clock, model_fields)

    # Advecting velocity: the field itself for momentum, the normal flow for tracers
    Uᵃ = advecting_velocity(U, ψ̄)

    # Phase speed: advecting velocity + gravity wave speed
    c = Uᵃ + c★
    Ũ = max(0, min(1, Δt / ΔX * c))

    # Inflow vs outflow relaxation
    τ = ifelse(Uᵃ >= 0, pa.outflow_timescale, pa.inflow_timescale)
    τ̃ = Δt / τ

    ψ_new = (ψᴮ + Ũ * ψᴬ + ψ̄ * τ̃) / (1 + τ̃ + Ũ)
    ψ_new = ifelse(τ == 0, ψ̄, ψ_new)

    # Convert back to extensive space (no-op when density is nothing)
    @inbounds ψ[iᴮ, jᴮ, kᴮ] = to_extensive(ρ, ψ_new, k)

    return nothing
end

@inline function step_left_open_boundary!(bc::PABC, l, m, boundary_indices, boundary_adjacent_indices,
                                          grid, ψ, U, clock, model_fields, ΔX, k)
    iᴮ, jᴮ, kᴮ = boundary_indices
    iᴬ, jᴬ, kᴬ = boundary_adjacent_indices
    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), zero(Δt), Δt)

    pa = bc.classification.scheme
    ρ = pa.density
    c★ = pa.gravity_wave_speed

    ψᴮ = to_intensive(ρ, @inbounds(ψ[iᴮ, jᴮ, kᴮ]), k)
    ψᴬ = to_intensive(ρ, @inbounds(ψ[iᴬ, jᴬ, kᴬ]), k)

    ψ̄ = getbc(bc, l, m, grid, clock, model_fields)

    # Advecting velocity: the field itself for momentum, the normal flow for tracers
    Uᵃ = advecting_velocity(U, ψ̄)

    # Phase speed: advecting velocity - gravity wave speed (outflow is -x at west / -y at south)
    c = Uᵃ - c★
    Ũ = min(0, max(-1, Δt / ΔX * c))

    τ = ifelse(Uᵃ <= 0, pa.outflow_timescale, pa.inflow_timescale)
    τ̃ = Δt / τ

    ψ_new = (ψᴮ - Ũ * ψᴬ + ψ̄ * τ̃) / (1 + τ̃ - Ũ)
    ψ_new = ifelse(τ == 0, ψ̄, ψ_new)

    @inbounds ψ[iᴮ, jᴮ, kᴮ] = to_extensive(ρ, ψ_new, k)

    return nothing
end

# Backward compatibility: old step_right/left_boundary! signatures without k argument
@inline function step_right_boundary!(bc::PABC, l, m, boundary_indices, boundary_adjacent_indices,
                                      grid, ψ, clock, model_fields, ΔX)
    k = boundary_indices[3]
    step_right_open_boundary!(bc, l, m, boundary_indices, boundary_adjacent_indices,
                              grid, ψ, nothing, clock, model_fields, ΔX, k)
end

@inline function step_left_boundary!(bc::PABC, l, m, boundary_indices, boundary_adjacent_indices,
                                     grid, ψ, clock, model_fields, ΔX)
    k = boundary_indices[3]
    step_left_open_boundary!(bc, l, m, boundary_indices, boundary_adjacent_indices,
                             grid, ψ, nothing, clock, model_fields, ΔX, k)
end

#####
##### Halo-filling methods for Face-located fields (velocity/momentum): NormalFlow + PerturbationAdvection
#####

@inline function _fill_east_halo!(j, k, grid, u, bc::PANFBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    i = grid.Nx + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i-1, j, k)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    step_right_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                              grid, u, nothing, clock, model_fields, Δx, k)
    return nothing
end

@inline function _fill_west_halo!(j, k, grid, u, bc::PANFBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    boundary_indices = (1, j, k)
    boundary_adjacent_indices = (2, j, k)
    Δx = Δxᶠᶜᶜ(1, j, k, grid)
    step_left_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                             grid, u, nothing, clock, model_fields, Δx, k)
    return nothing
end

@inline function _fill_north_halo!(i, k, grid, u, bc::PANFBC, ::Tuple{Any, Face, Any}, clock, model_fields)
    j = grid.Ny + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j-1, k)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    step_right_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                              grid, u, nothing, clock, model_fields, Δy, k)
    return nothing
end

@inline function _fill_south_halo!(i, k, grid, u, bc::PANFBC, ::Tuple{Any, Face, Any}, clock, model_fields)
    boundary_indices = (i, 1, k)
    boundary_adjacent_indices = (i, 2, k)
    Δy = Δyᶜᶠᶜ(i, 1, k, grid)
    step_left_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                             grid, u, nothing, clock, model_fields, Δy, k)
    return nothing
end

@inline function _fill_top_halo!(i, j, grid, u, bc::PANFBC, ::Tuple{Any, Any, Face}, clock, model_fields)
    k = grid.Nz + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j, k-1)
    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    step_right_open_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices,
                              grid, u, nothing, clock, model_fields, Δz, k)
    return nothing
end

@inline function _fill_bottom_halo!(i, j, grid, u, bc::PANFBC, ::Tuple{Any, Any, Face}, clock, model_fields)
    boundary_indices = (i, j, 1)
    boundary_adjacent_indices = (i, j, 2)
    Δz = Δzᶜᶜᶠ(i, j, 1, grid)
    step_left_open_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices,
                             grid, u, nothing, clock, model_fields, Δz, 1)
    return nothing
end

#####
##### Halo-filling methods for Center-located fields (scalars like ρθ, ρq, tracers): Value + PerturbationAdvection
#####

@inline function _fill_east_halo!(j, k, grid, c, bc::PAVBC, ::Tuple{Center, Any, Any}, clock, model_fields)
    i = grid.Nx + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i-1, j, k)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    U = @inbounds model_fields.u[i, j, k]
    step_right_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                              grid, c, U, clock, model_fields, Δx, k)
    return nothing
end

@inline function _fill_west_halo!(j, k, grid, c, bc::PAVBC, ::Tuple{Center, Any, Any}, clock, model_fields)
    boundary_indices = (0, j, k)
    boundary_adjacent_indices = (1, j, k)
    Δx = Δxᶠᶜᶜ(1, j, k, grid)
    U = @inbounds model_fields.u[1, j, k]
    step_left_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                             grid, c, U, clock, model_fields, Δx, k)
    return nothing
end

@inline function _fill_north_halo!(i, k, grid, c, bc::PAVBC, ::Tuple{Any, Center, Any}, clock, model_fields)
    j = grid.Ny + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j-1, k)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    U = @inbounds model_fields.v[i, j, k]
    step_right_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                              grid, c, U, clock, model_fields, Δy, k)
    return nothing
end

@inline function _fill_south_halo!(i, k, grid, c, bc::PAVBC, ::Tuple{Any, Center, Any}, clock, model_fields)
    boundary_indices = (i, 0, k)
    boundary_adjacent_indices = (i, 1, k)
    Δy = Δyᶜᶠᶜ(i, 1, k, grid)
    U = @inbounds model_fields.v[i, 1, k]
    step_left_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                             grid, c, U, clock, model_fields, Δy, k)
    return nothing
end

@inline function _fill_top_halo!(i, j, grid, c, bc::PAVBC, ::Tuple{Any, Any, Center}, clock, model_fields)
    k = grid.Nz + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j, k-1)
    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    U = @inbounds model_fields.w[i, j, k]
    step_right_open_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices,
                              grid, c, U, clock, model_fields, Δz, k)
    return nothing
end

@inline function _fill_bottom_halo!(i, j, grid, c, bc::PAVBC, ::Tuple{Any, Any, Center}, clock, model_fields)
    boundary_indices = (i, j, 0)
    boundary_adjacent_indices = (i, j, 1)
    Δz = Δzᶜᶜᶠ(i, j, 1, grid)
    U = @inbounds model_fields.w[i, j, 1]
    step_left_open_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices,
                             grid, c, U, clock, model_fields, Δz, 1)
    return nothing
end
