using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using Oceananigans.Grids: Center, Face
using Oceananigans: defaults
using Oceananigans.Utils: prettysummary

struct PerturbationAdvection{FT, D}
    inflow_timescale :: FT
    outflow_timescale :: FT
    gravity_wave_speed :: FT
    density :: D
end

"""
    PerturbationAdvection(FT = defaults.FloatType;
                          outflow_timescale = Inf,
                          inflow_timescale = 0,
                          gravity_wave_speed = 0,
                          density = nothing)

Create a `PerturbationAdvection` scheme to be used with an `OpenBoundaryCondition`.
This scheme nudges the boundary value to the `OpenBoundaryCondition`'s exterior value `val`,
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
`ψ` with prescribed exterior value `ψ̄`. Denoting the boundary value
as `ψᴮ` and the adjacent interior value as `ψᴬ`, and noting that the
perturbation is `ψ' = ψ - ψ̄`, we take a backwards Euler step
on a right boundary:

```math
(ψ̄ⁿ⁺¹ - ψ̄ⁿ) / Δt + (ψ'ᴮⁿ⁺¹ - ψ'ᴮⁿ) / Δt = -ψ̄ⁿ⁺¹ (ψ'ᴮⁿ⁺¹ - ψ'ᴬⁿ⁺¹) / Δx + Fψ
```

This cannot be solved for general forcing, but if we assume the dominant forcing is
relaxation to the exterior value (i.e. `ψ' → 0`) then `Fψ = -ψ' / τ`,
and we can find `ψ'ᴮⁿ⁺¹`:

```math
ψ'ᴮⁿ⁺¹ = (ψᴮⁿ + Ũ ψ'ᴬⁿ⁺¹ - ψ̄ⁿ⁺¹) / (1 + Ũ + Δt / τ)
```

where `Ũ = ψ̄ Δt / Δx`. Then `ψᴮⁿ⁺¹` is:

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
  prognostic ρu, ρθ).

  Accepts an `AbstractField` or a `FieldTimeSeries`. For an `AbstractField` ρ,
  the value is interpolated from ρ's location to ψ's location with standard
  staggered-grid operators (`ℑxᶠᵃᵃ` and similar). For a `FieldTimeSeries` ρ
  the value is interpolated in both space and time, so the FTS may live on a
  different grid than the simulation — useful for regional hindcasts where
  boundary density is diagnosed from reanalysis thermodynamics.

  Default: `nothing` (no conversion).
"""
function PerturbationAdvection(FT = defaults.FloatType;
                               outflow_timescale = Inf,
                               inflow_timescale = 0,
                               gravity_wave_speed = 0,
                               density = nothing)
    inflow_timescale = convert(FT, inflow_timescale)
    outflow_timescale = convert(FT, outflow_timescale)
    gravity_wave_speed = convert(FT, gravity_wave_speed)
    return PerturbationAdvection(inflow_timescale, outflow_timescale, gravity_wave_speed, density)
end

# Support 2-positional-arg constructor
PerturbationAdvection(inflow_timescale, outflow_timescale) =
    PerturbationAdvection(inflow_timescale, outflow_timescale, zero(inflow_timescale), nothing)

Adapt.adapt_structure(to, pe::PerturbationAdvection) =
    PerturbationAdvection(adapt(to, pe.inflow_timescale),
                          adapt(to, pe.outflow_timescale),
                          adapt(to, pe.gravity_wave_speed),
                          adapt(to, pe.density))

Base.summary(::PerturbationAdvection{FT}) where FT = "PerturbationAdvection{$FT}"

function Base.show(io::IO, pe::PerturbationAdvection)
    print(io, summary(pe), '\n')
    print(io, "├── inflow_timescale: ", prettysummary(pe.inflow_timescale), '\n')
    print(io, "├── outflow_timescale: ", prettysummary(pe.outflow_timescale), '\n')
    print(io, "├── gravity_wave_speed: ", prettysummary(pe.gravity_wave_speed), '\n')
    print(io, "└── density: ", prettysummary(pe.density))
end

const PAOBC = BoundaryCondition{<:Open{<:PerturbationAdvection}}

# Density value at ψ's grid location for a static `AbstractField` density.
# Staggered-grid interpolation is applied when ρ is at Centers and ψ is at a
# Face. Column-only density fields (e.g. `Field{Nothing, Nothing, Center}`
# for an anelastic reference profile) broadcast horizontally so the same
# path is correct for them.
#
# `FlavorOfFTS` densities are handled in `OutputReaders` via a separate
# overload of `to_intensive` / `to_extensive` defined after `interpolate`,
# `Time`, etc. are available.
@inline _pa_density_value(ρ, i, j, k, grid, ::Tuple{Face,   Center, Center}) = ℑxᶠᵃᵃ(i, j, k, grid, ρ)
@inline _pa_density_value(ρ, i, j, k, grid, ::Tuple{Center, Face,   Center}) = ℑyᵃᶠᵃ(i, j, k, grid, ρ)
@inline _pa_density_value(ρ, i, j, k, grid, ::Tuple{Center, Center, Face})   = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
@inline _pa_density_value(ρ, i, j, k, grid, ::Tuple{Center, Center, Center}) = @inbounds ρ[i, j, k]

# Helpers to convert between density-weighted and intensive fields.
# When density is `nothing`, these are no-ops. `to_intensive` reads ψ at
# `(i, j, k)` and divides by ρ at ψ's location; `to_extensive` takes the
# *value* (already in intensive space) and multiplies by ρ, with the caller
# writing the result back into `ψ`. `clock` is threaded through for the
# benefit of `FieldTimeSeries` densities (extended in `OutputReaders`).
@inline to_intensive(::Nothing, ψ, i, j, k, grid, loc, clock) = @inbounds ψ[i, j, k]
@inline to_intensive(ρ,         ψ, i, j, k, grid, loc, clock) =
    @inbounds ψ[i, j, k] / _pa_density_value(ρ, i, j, k, grid, loc)

@inline to_extensive(::Nothing, ψ_value, i, j, k, grid, loc, clock) = ψ_value
@inline to_extensive(ρ,         ψ_value, i, j, k, grid, loc, clock) =
    _pa_density_value(ρ, i, j, k, grid, loc) * ψ_value

@inline function step_right_open_boundary!(bc::PAOBC, l, m, boundary_indices, boundary_adjacent_indices,
                                           grid, ψ, clock, model_fields, ΔX, loc)
    iᴮ, jᴮ, kᴮ = boundary_indices
    iᴬ, jᴬ, kᴬ = boundary_adjacent_indices
    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), zero(Δt), Δt)

    pa = bc.classification.scheme
    ρ = pa.density
    c★ = pa.gravity_wave_speed

    # Convert to intensive space (no-op when density is nothing)
    ψᴮ = to_intensive(ρ, ψ, iᴮ, jᴮ, kᴮ, grid, loc, clock)
    ψᴬ = to_intensive(ρ, ψ, iᴬ, jᴬ, kᴬ, grid, loc, clock)

    # Prescribed exterior value (in intensive units when density is provided)
    ψ̄ = getbc(bc, l, m, grid, clock, model_fields)

    # Phase speed: exterior value + gravity wave speed
    c = ψ̄ + c★
    Ũ = max(0, min(1, Δt / ΔX * c))

    # Inflow vs outflow relaxation
    τ = ifelse(ψ̄ >= 0, pa.outflow_timescale, pa.inflow_timescale)
    τ̃ = Δt / τ

    ψ_new = (ψᴮ + Ũ * ψᴬ + ψ̄ * τ̃) / (1 + τ̃ + Ũ)
    ψ_new = ifelse(τ == 0, ψ̄, ψ_new)

    # Convert back to extensive space (no-op when density is nothing)
    @inbounds ψ[iᴮ, jᴮ, kᴮ] = to_extensive(ρ, ψ_new, iᴮ, jᴮ, kᴮ, grid, loc, clock)

    return nothing
end

@inline function step_left_open_boundary!(bc::PAOBC, l, m, boundary_indices, boundary_adjacent_indices,
                                          grid, ψ, clock, model_fields, ΔX, loc)
    iᴮ, jᴮ, kᴮ = boundary_indices
    iᴬ, jᴬ, kᴬ = boundary_adjacent_indices
    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), zero(Δt), Δt)

    pa = bc.classification.scheme
    ρ = pa.density
    c★ = pa.gravity_wave_speed

    ψᴮ = to_intensive(ρ, ψ, iᴮ, jᴮ, kᴮ, grid, loc, clock)
    ψᴬ = to_intensive(ρ, ψ, iᴬ, jᴬ, kᴬ, grid, loc, clock)

    ψ̄ = getbc(bc, l, m, grid, clock, model_fields)

    # Phase speed: exterior value - gravity wave speed (outflow is -x at west / -y at south)
    c = ψ̄ - c★
    Ũ = min(0, max(-1, Δt / ΔX * c))

    τ = ifelse(ψ̄ <= 0, pa.outflow_timescale, pa.inflow_timescale)
    τ̃ = Δt / τ

    ψ_new = (ψᴮ - Ũ * ψᴬ + ψ̄ * τ̃) / (1 + τ̃ - Ũ)
    ψ_new = ifelse(τ == 0, ψ̄, ψ_new)

    @inbounds ψ[iᴮ, jᴮ, kᴮ] = to_extensive(ρ, ψ_new, iᴮ, jᴮ, kᴮ, grid, loc, clock)

    return nothing
end

# Aliases for callers that follow the generic boundary-step naming.
# Default to a Center-located ψ so the existing column-density behavior
# (direct `ρ[i, j, k]` indexing) is preserved for any external caller.
@inline step_right_boundary!(bc::PAOBC, l, m, boundary_indices, boundary_adjacent_indices,
                             grid, ψ, clock, model_fields, ΔX) =
    step_right_open_boundary!(bc, l, m, boundary_indices, boundary_adjacent_indices,
                              grid, ψ, clock, model_fields, ΔX,
                              (Center(), Center(), Center()))

@inline step_left_boundary!(bc::PAOBC, l, m, boundary_indices, boundary_adjacent_indices,
                            grid, ψ, clock, model_fields, ΔX) =
    step_left_open_boundary!(bc, l, m, boundary_indices, boundary_adjacent_indices,
                             grid, ψ, clock, model_fields, ΔX,
                             (Center(), Center(), Center()))

#####
##### Halo-filling methods for Face-located fields (velocity/momentum)
#####

@inline function _fill_east_halo!(j, k, grid, u, bc::PAOBC, loc::Tuple{Face, Any, Any}, clock, model_fields)
    i = grid.Nx + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i-1, j, k)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    step_right_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                              grid, u, clock, model_fields, Δx, loc)
    return nothing
end

@inline function _fill_west_halo!(j, k, grid, u, bc::PAOBC, loc::Tuple{Face, Any, Any}, clock, model_fields)
    boundary_indices = (1, j, k)
    boundary_adjacent_indices = (2, j, k)
    Δx = Δxᶠᶜᶜ(1, j, k, grid)
    step_left_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                             grid, u, clock, model_fields, Δx, loc)
    return nothing
end

@inline function _fill_north_halo!(i, k, grid, u, bc::PAOBC, loc::Tuple{Any, Face, Any}, clock, model_fields)
    j = grid.Ny + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j-1, k)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    step_right_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                              grid, u, clock, model_fields, Δy, loc)
    return nothing
end

@inline function _fill_south_halo!(i, k, grid, u, bc::PAOBC, loc::Tuple{Any, Face, Any}, clock, model_fields)
    boundary_indices = (i, 1, k)
    boundary_adjacent_indices = (i, 2, k)
    Δy = Δyᶜᶠᶜ(i, 1, k, grid)
    step_left_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                             grid, u, clock, model_fields, Δy, loc)
    return nothing
end

@inline function _fill_top_halo!(i, j, grid, u, bc::PAOBC, loc::Tuple{Any, Any, Face}, clock, model_fields)
    k = grid.Nz + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j, k-1)
    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    step_right_open_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices,
                              grid, u, clock, model_fields, Δz, loc)
    return nothing
end

@inline function _fill_bottom_halo!(i, j, grid, u, bc::PAOBC, loc::Tuple{Any, Any, Face}, clock, model_fields)
    boundary_indices = (i, j, 1)
    boundary_adjacent_indices = (i, j, 2)
    Δz = Δzᶜᶜᶠ(i, j, 1, grid)
    step_left_open_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices,
                             grid, u, clock, model_fields, Δz, loc)
    return nothing
end

#####
##### Halo-filling methods for Center-located fields (scalars like ρθ, ρq, tracers)
#####

@inline function _fill_east_halo!(j, k, grid, c, bc::PAOBC, loc::Tuple{Center, Any, Any}, clock, model_fields)
    i = grid.Nx + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i-1, j, k)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    step_right_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                              grid, c, clock, model_fields, Δx, loc)
    return nothing
end

@inline function _fill_west_halo!(j, k, grid, c, bc::PAOBC, loc::Tuple{Center, Any, Any}, clock, model_fields)
    boundary_indices = (1, j, k)
    boundary_adjacent_indices = (2, j, k)
    Δx = Δxᶠᶜᶜ(1, j, k, grid)
    step_left_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                             grid, c, clock, model_fields, Δx, loc)
    return nothing
end

@inline function _fill_north_halo!(i, k, grid, c, bc::PAOBC, loc::Tuple{Any, Center, Any}, clock, model_fields)
    j = grid.Ny + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j-1, k)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    step_right_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                              grid, c, clock, model_fields, Δy, loc)
    return nothing
end

@inline function _fill_south_halo!(i, k, grid, c, bc::PAOBC, loc::Tuple{Any, Center, Any}, clock, model_fields)
    boundary_indices = (i, 1, k)
    boundary_adjacent_indices = (i, 2, k)
    Δy = Δyᶜᶠᶜ(i, 1, k, grid)
    step_left_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                             grid, c, clock, model_fields, Δy, loc)
    return nothing
end
