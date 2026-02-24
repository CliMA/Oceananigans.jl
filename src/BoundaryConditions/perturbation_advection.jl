using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ
using Oceananigans: defaults

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
\\partial_t u + u \\cdot \\nabla u = -\\nabla P + F,
```

and note that on the boundary the pressure gradient is zero.
We can then assume that the flow composes of mean (``\\vec{U}``) and perturbation (``\\vec{u'}``) components,
and considering the ``x``-component of velocity, we can rewrite the equation as

```math
\\partial_t u_1 = -u_1 \\partial_1 u - u_2 \\partial_2 u_1 - u_3 \\partial_3 u_1 + F_1
              \\approx -U_1 \\partial_1 u_1' - U_2 \\partial_2 u_1' - U_3 \\partial_3 u_1' + F.
```

Simplify by assuming that ``\\vec{U} = U \\hat{x}``, and then take a numerical step to find ``u_1``.
While derived for velocity, the resulting scheme generalizes to any prognostic field
``\\psi`` with prescribed exterior value ``\\bar{\\psi}``. Denoting the boundary value
as ``\\psi^B`` and the adjacent interior value as ``\\psi^A``, and noting that the
perturbation is ``\\psi' = \\psi - \\bar{\\psi}``, we take a backwards Euler step
on a right boundary:

```math
\\frac{\\bar{\\psi}^{n+1} - \\bar{\\psi}^n}{\\Delta t}
    + \\frac{\\psi'^{B, n+1} - \\psi'^{B, n}}{\\Delta t}
    = -\\bar{\\psi}^{n+1}
      \\frac{\\psi'^{B, n+1} - \\psi'^{A, n+1}}{\\Delta x} + F_\\psi.
```

This cannot be solved for general forcing, but if we assume the dominant forcing is
relaxation to the exterior value (i.e. ``\\psi' \\to 0``) then ``F_\\psi = -\\psi' / \\tau``,
and we can find ``\\psi'^{B, n+1}``:

```math
\\psi'^{B, n+1} = \\frac{\\psi^{B, n} + \\tilde{U} \\, \\psi'^{A, n+1} - \\bar{\\psi}^{n+1}}
                       {1 + \\tilde{U} + \\Delta t / \\tau},
```

where ``\\tilde{U} = \\bar{\\psi} \\, \\Delta t / \\Delta x``. Then ``\\psi^{B, n+1}`` is:

```math
\\psi^{B, n+1} = \\frac{\\psi^{B, n} + \\tilde{U} \\, \\psi^{A, n+1} + \\bar{\\psi}^{n+1} \\tilde{\\tau}}
                      {1 + \\tilde{\\tau} + \\tilde{U}}
```

where ``\\tilde{\\tau} = \\Delta t / \\tau``.

The same operation can be repeated for left boundaries.

The relaxation timescale ``τ`` can be set to different values depending on whether
``U`` points in or out of the domain (`inflow_timescale`/`outflow_timescale`). Since the
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

const PAOBC = BoundaryCondition{<:Open{<:PerturbationAdvection}}

# Helper to convert between density-weighted and intensive fields.
# When density is nothing, these are no-ops.
@inline to_intensive(::Nothing, ψ, k) = ψ
@inline to_intensive(ρ, ψ, k) = ψ / @inbounds ρ[1, 1, k]
@inline to_extensive(::Nothing, ψ, k) = ψ
@inline to_extensive(ρ, ψ, k) = @inbounds ρ[1, 1, k] * ψ

@inline function step_right_open_boundary!(bc::PAOBC, l, m, boundary_indices, boundary_adjacent_indices,
                                           grid, ψ, clock, model_fields, ΔX, k)
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

    # Phase speed: exterior value + gravity wave speed
    c = ψ̄ + c★
    Ũ = max(0, min(1, Δt / ΔX * c))

    # Inflow vs outflow relaxation
    τ = ifelse(ψ̄ >= 0, pa.outflow_timescale, pa.inflow_timescale)
    τ̃ = Δt / τ

    ψ_new = (ψᴮ + Ũ * ψᴬ + ψ̄ * τ̃) / (1 + τ̃ + Ũ)
    ψ_new = ifelse(τ == 0, ψ̄, ψ_new)

    # Convert back to extensive space (no-op when density is nothing)
    @inbounds ψ[iᴮ, jᴮ, kᴮ] = to_extensive(ρ, ψ_new, k)

    return nothing
end

@inline function step_left_open_boundary!(bc::PAOBC, l, m, boundary_indices, boundary_adjacent_indices,
                                          grid, ψ, clock, model_fields, ΔX, k)
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

    # Phase speed: exterior value - gravity wave speed (outflow is -x at west / -y at south)
    c = ψ̄ - c★
    Ũ = min(0, max(-1, Δt / ΔX * c))

    τ = ifelse(ψ̄ <= 0, pa.outflow_timescale, pa.inflow_timescale)
    τ̃ = Δt / τ

    ψ_new = (ψᴮ - Ũ * ψᴬ + ψ̄ * τ̃) / (1 + τ̃ - Ũ)
    ψ_new = ifelse(τ == 0, ψ̄, ψ_new)

    @inbounds ψ[iᴮ, jᴮ, kᴮ] = to_extensive(ρ, ψ_new, k)

    return nothing
end

# Backward compatibility: old step_right/left_boundary! signatures without k argument
@inline function step_right_boundary!(bc::PAOBC, l, m, boundary_indices, boundary_adjacent_indices,
                                      grid, ψ, clock, model_fields, ΔX)
    k = boundary_indices[3]
    step_right_open_boundary!(bc, l, m, boundary_indices, boundary_adjacent_indices,
                              grid, ψ, clock, model_fields, ΔX, k)
end

@inline function step_left_boundary!(bc::PAOBC, l, m, boundary_indices, boundary_adjacent_indices,
                                     grid, ψ, clock, model_fields, ΔX)
    k = boundary_indices[3]
    step_left_open_boundary!(bc, l, m, boundary_indices, boundary_adjacent_indices,
                             grid, ψ, clock, model_fields, ΔX, k)
end

#####
##### Halo-filling methods for Face-located fields (velocity/momentum)
#####

@inline function _fill_east_halo!(j, k, grid, u, bc::PAOBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    i = grid.Nx + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i-1, j, k)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    step_right_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                              grid, u, clock, model_fields, Δx, k)
    return nothing
end

@inline function _fill_west_halo!(j, k, grid, u, bc::PAOBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    boundary_indices = (1, j, k)
    boundary_adjacent_indices = (2, j, k)
    Δx = Δxᶠᶜᶜ(1, j, k, grid)
    step_left_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                             grid, u, clock, model_fields, Δx, k)
    return nothing
end

@inline function _fill_north_halo!(i, k, grid, u, bc::PAOBC, ::Tuple{Any, Face, Any}, clock, model_fields)
    j = grid.Ny + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j-1, k)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    step_right_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                              grid, u, clock, model_fields, Δy, k)
    return nothing
end

@inline function _fill_south_halo!(i, k, grid, u, bc::PAOBC, ::Tuple{Any, Face, Any}, clock, model_fields)
    boundary_indices = (i, 1, k)
    boundary_adjacent_indices = (i, 2, k)
    Δy = Δyᶜᶠᶜ(i, 1, k, grid)
    step_left_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                             grid, u, clock, model_fields, Δy, k)
    return nothing
end

@inline function _fill_top_halo!(i, j, grid, u, bc::PAOBC, ::Tuple{Any, Any, Face}, clock, model_fields)
    k = grid.Nz + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j, k-1)
    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    step_right_open_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices,
                              grid, u, clock, model_fields, Δz, k)
    return nothing
end

@inline function _fill_bottom_halo!(i, j, grid, u, bc::PAOBC, ::Tuple{Any, Any, Face}, clock, model_fields)
    boundary_indices = (i, j, 1)
    boundary_adjacent_indices = (i, j, 2)
    Δz = Δzᶜᶜᶠ(i, j, 1, grid)
    step_left_open_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices,
                             grid, u, clock, model_fields, Δz, 1)
    return nothing
end

#####
##### Halo-filling methods for Center-located fields (scalars like ρθ, ρq, tracers)
#####

@inline function _fill_east_halo!(j, k, grid, c, bc::PAOBC, ::Tuple{Center, Any, Any}, clock, model_fields)
    i = grid.Nx + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i-1, j, k)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    step_right_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                              grid, c, clock, model_fields, Δx, k)
    return nothing
end

@inline function _fill_west_halo!(j, k, grid, c, bc::PAOBC, ::Tuple{Center, Any, Any}, clock, model_fields)
    boundary_indices = (1, j, k)
    boundary_adjacent_indices = (2, j, k)
    Δx = Δxᶠᶜᶜ(1, j, k, grid)
    step_left_open_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices,
                             grid, c, clock, model_fields, Δx, k)
    return nothing
end

@inline function _fill_north_halo!(i, k, grid, c, bc::PAOBC, ::Tuple{Any, Center, Any}, clock, model_fields)
    j = grid.Ny + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j-1, k)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    step_right_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                              grid, c, clock, model_fields, Δy, k)
    return nothing
end

@inline function _fill_south_halo!(i, k, grid, c, bc::PAOBC, ::Tuple{Any, Center, Any}, clock, model_fields)
    boundary_indices = (i, 1, k)
    boundary_adjacent_indices = (i, 2, k)
    Δy = Δyᶜᶠᶜ(i, 1, k, grid)
    step_left_open_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices,
                             grid, c, clock, model_fields, Δy, k)
    return nothing
end
