#####
##### ObliqueRadiation (Raymond & Kuo 1984) open boundary scheme
#####

"""
    ObliqueRadiation(; inflow_timescale = 0,
                       outflow_timescale = Inf,
                       use_boundary_velocity = false,
                       target_transport = nothing)

[Raymond & Kuo (1984)](@cite raymond.kuo1984) two-dimensional radiation condition with adaptive nudging
([Marchesiello et al., 2001](@cite marchesiello.ea2001)):

    ∂φ/∂t + cₙ ∂φ/∂n + cₜ ∂φ/∂t̂ = - (φ - φᵉˣᵗ) / τ

with

    cₙ = -(∂φ/∂t)(∂φ/∂n) / |∇φ|²,   cₜ = -(∂φ/∂t)(∂φ/∂t̂) / |∇φ|²,   |∇φ|² = (∂φ/∂n)² + (∂φ/∂t̂)²

where `n` is the boundary-normal and `t̂` the horizontal boundary-tangential direction. The
tangential derivative is upwinded and `cₜ` is limited to a tangential Courant number of one.
For `∂φ/∂t̂ = 0` the scheme reduces to `NormalRadiation`. The tangential term is applied on the
lateral boundaries only.

Inflow versus outflow is decided from the boundary-normal velocity: on inflow `cₙ = cₜ = 0` and
`τ = inflow_timescale`; on outflow `τ = outflow_timescale`. For `Value` boundary conditions,
`use_boundary_velocity = true` takes that velocity at the boundary face rather than one cell into
the interior. `target_transport` pins the net transport through a `NormalFlowBoundaryCondition`, as for
[`NormalRadiation`](@ref).

References
==========
* Raymond, W. H., & Kuo, H. L. (1984). "A radiation boundary condition for multi-dimensional
  flows." Quarterly Journal of the Royal Meteorological Society, 110(464), 535-551.
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
├── use_boundary_velocity: false
└── target_transport: nothing
```
"""
struct ObliqueRadiation{FT, S, TF} <: AbstractRadiationScheme{FT}
    outflow_timescale :: FT
    inflow_timescale  :: FT
    use_boundary_velocity :: Bool
    φ⁻ :: S
    φ₁ :: S
    target_transport :: TF # prescribed net transport through the boundary, or nothing
    periodic :: Bool       # whether the direction along the boundary is periodic on this grid
end

function ObliqueRadiation(FT = defaults.FloatType;
                          inflow_timescale = 0,
                          outflow_timescale = Inf,
                          use_boundary_velocity = false,
                          target_transport = nothing)

    outflow_timescale = convert(FT, outflow_timescale)
    inflow_timescale = convert(FT, inflow_timescale)
    target_transport = convert_target_transport(FT, target_transport)
    return ObliqueRadiation(outflow_timescale, inflow_timescale, use_boundary_velocity, nothing, nothing, target_transport, false)
end

Adapt.adapt_structure(to, r::ObliqueRadiation) =
    ObliqueRadiation(adapt(to, r.outflow_timescale),
                     adapt(to, r.inflow_timescale),
                     r.use_boundary_velocity,
                     adapt(to, r.φ⁻),
                     adapt(to, r.φ₁),
                     adapt(to, r.target_transport),
                     r.periodic)

has_target_transport(::ObliqueRadiation{<:Any, <:Any, <:Nothing}) = false
has_target_transport(::ObliqueRadiation) = true

radiation_storage(radiation::ObliqueRadiation, buffer) =
    ObliqueRadiation(radiation.outflow_timescale, radiation.inflow_timescale, radiation.use_boundary_velocity,
                     buffer(), buffer(), radiation.target_transport, radiation.periodic)

# The buffers span the Connected halos along the boundary, but a periodic direction that is not partitioned wraps
# onto itself.
function materialize_radiation_storage(radiation::ObliqueRadiation, grid, loc, dim)
    scheme = invoke(materialize_radiation_storage, Tuple{AbstractRadiationScheme, Any, Any, Any}, radiation, grid, loc, dim)
    periodic = topology(grid, ifelse(dim == 1, 2, 1)) === Grids.Periodic
    return ObliqueRadiation(scheme.outflow_timescale, scheme.inflow_timescale, scheme.use_boundary_velocity,
                            scheme.φ⁻, scheme.φ₁, scheme.target_transport, periodic)
end

# Backward and forward differences along the boundary face. Beyond its ends they are zero, or take the value from the
# other end if the direction is periodic.
@inline function tangential_differences(φ, i, j, periodic)
    lower = firstindex(φ, 1)
    upper = lastindex(φ, 1)
    i₋ = ifelse(periodic, ifelse(i == lower, upper, i - 1), max(i - 1, lower))
    i₊ = ifelse(periodic, ifelse(i == upper, lower, i + 1), min(i + 1, upper))
    @inbounds begin
        φ₀ = φ[i, j]
        φ₋ = φ[i₋, j]
        φ₊ = φ[i₊, j]
    end
    return φ₀ - φ₋, φ₊ - φ₀
end

@inline function oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δᵇ₋, δᵇ₊, δ₁₋, δ₁₊, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
    ∂t_φ = φ₁ⁿ⁺¹ - φ₁ⁿ
    ∂ξ_φ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹
    ∂η_φ = ifelse(-∂t_φ * (δ₁₋ + δ₁₊) > 0, δ₁₋, δ₁₊)

    FT = typeof(∂ξ_φ)
    D  = max(∂ξ_φ^2 + ∂η_φ^2, eps(FT))
    Cᶜ = - ∂t_φ * ∂ξ_φ / D
    Cₙ = ifelse(outflow, max(zero(FT), min(one(FT), max(Cᶜ, Cᵃ))), zero(FT))
    Cₜ = ifelse(outflow, clamp(- ∂t_φ * ∂η_φ / D, -one(FT), one(FT)), zero(FT))

    τ = ifelse(outflow, radiation.outflow_timescale, radiation.inflow_timescale)
    τ̃ = Δt / τ

    φᵇⁿ⁺¹ = (φᵇⁿ + Cₙ * φ₁ⁿ⁺¹ - max(Cₜ, zero(FT)) * δᵇ₋ - min(Cₜ, zero(FT)) * δᵇ₊ + τ̃ * φᵉˣᵗ) / (1 + Cₙ + τ̃)

    return ifelse(τ == 0, φᵉˣᵗ, φᵇⁿ⁺¹)
end

# The tangential derivatives are those of the boundary and interior values at the base of the stage
@inline function radiation_update(radiation::ObliqueRadiation, ::Union{West, East, South, North}, i, j, φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, outflow, Cᵃ)
    δᵇ₋, δᵇ₊ = tangential_differences(radiation.φ⁻, i, j, radiation.periodic)
    δ₁₋, δ₁₊ = tangential_differences(radiation.φ₁, i, j, radiation.periodic)
    return oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δᵇ₋, δᵇ₊, δ₁₋, δ₁₊, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
end
