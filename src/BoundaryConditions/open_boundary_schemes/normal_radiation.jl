#####
##### NormalRadiation (based on Orlanski 1976) open boundary scheme
#####

"""
    NormalRadiation(; inflow_timescale = 0,
                      outflow_timescale = Inf,
                      use_boundary_velocity = false,
                      target_transport = nothing)

[Orlanski (1976)](@cite Orlanksi1976) radiation condition with locally-diagnosed phase speed
and adaptive nudging ([Marchesiello et al., 2001](@cite marchesiello.ea2001)):

    ∂φ/∂t + cₙ ⋅ ∂φ/∂n = - (φ - φᵉˣᵗ) / τ

where `cₙ = −(∂φ/∂t) / (∂φ/∂n)` is diagnosed from interior values, clamped to `[0, Δx/Δt]`.
Inflow vs outflow is decided from the boundary-normal velocity: for `NormalFlow` fields the boundary
value itself, for `Value` fields (tracers, tangential velocities) a `model_fields` velocity taken one
cell into the interior, or at the boundary face when `use_boundary_velocity = true` (the interior value
is a prognostic velocity, the boundary value is itself radiation-extrapolated; see Orlanski 1976 and the
MITgcm `obcs` implementation). `τ = τ_in` and `cₙ = 0` on inflow, `τ = τ_out` on outflow.

`use_boundary_velocity` only affects `Value` boundary conditions (tracers, tangential velocities). It is
ignored for a `NormalFlowBoundaryCondition`, which radiates the normal velocity using its own boundary value.

`target_transport` pins the net volume transport through the boundary, the integral of the boundary-normal
velocity over the boundary area `∮u·dA` [m³ s⁻¹] taken positive in the positive coordinate direction, to a
prescribed value. It may be a number or a callable of the grid and only applies to a `NormalFlowBoundaryCondition`.
Each time step the boundary-normal velocity is shifted uniformly so that its transport matches the target; in a
`NonhydrostaticModel` the remaining net imbalance is then distributed over the open boundaries without a target.
The default `nothing` leaves the boundary in that global pool correction. See also [`PerturbationAdvection`](@ref),
which accepts the same keyword.

`NormalRadiation` is used as the `scheme` of a [`ValueBoundaryCondition`](@ref) for `Center`-located
fields such as tracers — where it updates the halo cell adjacent to the boundary — or of a
[`NormalFlowBoundaryCondition`](@ref) for boundary-normal velocities — where it updates the
boundary face value. Interior cells are never written.

The interior value at the base of the stage, needed by the Orlanski phase-speed diagnosis, is
stored in an array (`φ₁`) allocated automatically during boundary condition regularization.

References
==========
* Orlanski, I. (1976). "A simple boundary condition for unbounded hyperbolic flows."
  Journal of Computational Physics, 21(3), 251-269.
* Marchesiello, P., McWilliams, J. C., & Shchepetkin, A. (2001). "Open boundary conditions
  for long-term integration of regional oceanic models." Ocean Modelling, 3(1-2), 1-20.

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: NormalRadiation

NormalRadiation(outflow_timescale = 360 * 86400, inflow_timescale = 86400)

# output
NormalRadiation{Float64}
├── inflow_timescale: 86400.0
├── outflow_timescale: 3.1104e7
├── use_boundary_velocity: false
└── target_transport: nothing
```
"""
struct NormalRadiation{FT, S, TF} <: AbstractRadiationScheme{FT}
    outflow_timescale :: FT
    inflow_timescale  :: FT
    use_boundary_velocity :: Bool # advect Value fields with the boundary-face velocity instead of one cell in
    φ⁻  :: S  # boundary value at the base of the stage (2D array or nothing)
    φ₁  :: S  # interior value at the base of the stage (2D array or nothing)
    target_transport :: TF # prescribed net transport through the boundary, or nothing
end

function NormalRadiation(FT = defaults.FloatType;
                   inflow_timescale = 0,
                   outflow_timescale = Inf,
                   use_boundary_velocity = false,
                   target_transport = nothing)

    outflow_timescale = convert(FT, outflow_timescale)
    inflow_timescale = convert(FT, inflow_timescale)
    target_transport = convert_target_transport(FT, target_transport)
    return NormalRadiation(outflow_timescale, inflow_timescale, use_boundary_velocity, nothing, nothing, target_transport)
end

Adapt.adapt_structure(to, r::NormalRadiation) =
    NormalRadiation(adapt(to, r.outflow_timescale),
              adapt(to, r.inflow_timescale),
              r.use_boundary_velocity,
              adapt(to, r.φ⁻),
              adapt(to, r.φ₁),
              adapt(to, r.target_transport))

Base.summary(r::AbstractRadiationScheme{FT}) where FT = string(nameof(typeof(r)), "{$FT}")

function Base.show(io::IO, r::AbstractRadiationScheme)
    print(io, summary(r), '\n')
    print(io, "├── inflow_timescale: ",  prettysummary(r.inflow_timescale), '\n')
    print(io, "├── outflow_timescale: ", prettysummary(r.outflow_timescale), '\n')
    print(io, "├── use_boundary_velocity: ", r.use_boundary_velocity, '\n')
    print(io, "└── target_transport: ", prettysummary(r.target_transport))
end

has_target_transport(::NormalRadiation{<:Any, <:Any, <:Nothing}) = false
has_target_transport(::NormalRadiation) = true

get_target_transport(scheme::AbstractRadiationScheme, grid) = _eval_tt(scheme.target_transport, grid)
get_target_transport(scheme::AbstractRadiationScheme) = scheme.target_transport

radiation_storage(radiation::NormalRadiation, buffer) =
    NormalRadiation(radiation.outflow_timescale, radiation.inflow_timescale, radiation.use_boundary_velocity,
                    buffer(), buffer(), radiation.target_transport)

#####
##### NormalRadiation — Orlanski (1976) with Marchesiello et al. (2001) nudging
#####

# Inflow/outflow from boundary-normal velocity, not phase-speed sign: avoids extremum-exit blowup where ∂φ/∂ξ → 0.

@inline function normal_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
    # Diagnose phase speed Courant number (Orlanski 1976)
    ∂t_φ = φ₁ⁿ⁺¹ - φ₁ⁿ
    ∂ξ_φ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹

    # guard ∂ξ_φ == 0 (extremum at boundary face)
    Cᶜ = ifelse(∂ξ_φ == 0, zero(∂t_φ), - ∂t_φ / ∂ξ_φ)

    Cₙ = ifelse(outflow, max(zero(Cᶜ), min(one(Cᶜ), max(Cᶜ, Cᵃ))), zero(Cᶜ))

    τ = ifelse(outflow, radiation.outflow_timescale, radiation.inflow_timescale)
    τ̃ = Δt / τ

    φᵇⁿ⁺¹ = (φᵇⁿ + Cₙ * φ₁ⁿ⁺¹ + τ̃ * φᵉˣᵗ) / (1 + Cₙ + τ̃)

    return ifelse(τ == 0, φᵉˣᵗ, φᵇⁿ⁺¹)
end

@inline radiation_update(radiation, side, i, j, φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, outflow, Cᵃ) =
    normal_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)

const ORBC = Union{BoundaryCondition{<:Value{<:Union{NormalRadiation, ObliqueRadiation}}},
                   BoundaryCondition{<:NormalFlow{<:Union{NormalRadiation, ObliqueRadiation}}}}

# The Orlanski phase speed is diagnosed from the change of the first interior value over the stage
@inline function cache_exterior_state!(side, i, j, grid, c, bc::ORBC, loc, clock, model_fields)
    iᵇ, jᵇ, kᵇ = boundary_point(side, loc, grid, 0, i, j)
    i₁, j₁, k₁ = boundary_point(side, loc, grid, 1, i, j)
    radiation = bc.classification.scheme
    @inbounds radiation.φ⁻[i, j] = c[iᵇ, jᵇ, kᵇ]
    @inbounds radiation.φ₁[i, j] = c[i₁, j₁, k₁]
    return nothing
end

# A convex combination of the boundary value at the base of the stage, the interior and the exterior value, so it is bounded
@inline function step_exterior_state!(side, i, j, grid, c, bc::ORBC, loc, clock, model_fields, Δt)
    iᵇ, jᵇ, kᵇ = boundary_point(side, loc, grid, 0, i, j)
    i₁, j₁, k₁ = boundary_point(side, loc, grid, 1, i, j)
    i₂, j₂, k₂ = boundary_point(side, loc, grid, 2, i, j)
    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, j, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i₁, j₁, k₁]
        φ₂ⁿ⁺¹ = c[i₂, j₂, k₂]

        n  = ifelse(radiation.use_boundary_velocity, 0, 1)
        Uᵃ = advecting_velocity(bc, side, model_fields, grid, n, i, j, φ₁ⁿ⁺¹)
        Cᵃ = abs(Uᵃ) * Δt * reciprocal_boundary_spacing(side, grid, i, j)
        outflow = outward(side, Uᵃ) >= 0

        φᵇⁿ⁺¹ = radiation_update(radiation, side, i, j, radiation.φ⁻[i, j], φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, radiation.φ₁[i, j], φᵉˣᵗ, Δt, outflow, Cᵃ)
        c[iᵇ, jᵇ, kᵇ] = ifelse(immersed_boundary_point(side, loc, grid, i, j), zero(grid), φᵇⁿ⁺¹)
    end

    return nothing
end
