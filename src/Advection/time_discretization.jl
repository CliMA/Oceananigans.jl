abstract type AbstractTimeDiscretization end

"""
    struct ExplicitTimeDiscretization <: AbstractTimeDiscretization

A fully-explicit time-discretization.
"""
struct ExplicitTimeDiscretization <: AbstractTimeDiscretization end

Base.summary(::ExplicitTimeDiscretization) = "ExplicitTimeDiscretization"

"""
    struct VerticallyImplicitTimeDiscretization <: AbstractTimeDiscretization

A vertically-implicit time-discretization.

This implies that a flux divergence such as ``𝛁 ⋅ 𝐪`` at the ``n``-th timestep is
time-discretized as

```julia
[∇ ⋅ q]ⁿ = [explicit_flux_divergence]ⁿ + [∂z (κ ∂z c)]ⁿ⁺¹
```
"""
struct VerticallyImplicitTimeDiscretization <: AbstractTimeDiscretization end

Base.summary(::VerticallyImplicitTimeDiscretization) = "VerticallyImplicitTimeDiscretization"

struct AdaptiveVerticallyImplicitDiscretization{FT, R} <: AbstractTimeDiscretization
    cfl :: FT
    Δt  :: R
end

"""
    AdaptiveVerticallyImplicitDiscretization([FT = Oceananigans.defaults.FloatType]; cfl = 0.5)

An adaptively implicit vertical discretization scheme following Shchepetkin (2015) / CROCO.

Splits vertical advection into explicit and implicit parts based on the local vertical Courant number
`α = |w| Δt / Δz`. When `α ≤ cfl`, advection is fully explicit using `explicit_scheme`. When `α > cfl`,
the vertical velocity is decomposed as `w = wᵉ + wⁱ` where `wᵉ` is CFL-limited and `wⁱ` is treated
with implicit first-order upwind in the existing tridiagonal solver.

The splitting function is:

    f(α, cfl) = max(1, α / cfl)
    wᵉ = w / f  (explicit, CFL-limited)
    wⁱ = w - wᵉ (implicit, first-order upwind)

Keyword Arguments
=================

- `cfl`: Maximum vertical CFL for the explicit part (default: `0.5`).

Example
=======

```jldoctest
julia> using Oceananigans

julia> AdaptiveVerticallyImplicitDiscretization(cfl=0.3)
AdaptiveVerticallyImplicitDiscretization:
└── cfl: 0.3
```
"""
function AdaptiveVerticallyImplicitDiscretization(FT::DataType = Oceananigans.defaults.FloatType; cfl = 0.5)
    cfl = convert(FT, cfl)
    Δt  = Ref(zero(FT))
    return AdaptiveVerticallyImplicitDiscretization(cfl, Δt)
end

Adapt.adapt_structure(to, a::AdaptiveVerticallyImplicitDiscretization) =
    AdaptiveVerticallyImplicitDiscretization(a.cfl, a.Δt[])

Base.summary(a::AdaptiveVerticallyImplicitDiscretization) =
    string("AdaptiveVerticallyImplicitDiscretization(cfl=$(a.cfl))")

Base.show(io::IO, a::AdaptiveVerticallyImplicitDiscretization) =
    print(io, "AdaptiveVerticallyImplicitDiscretization:", "\n",
              "└── cfl: ", a.cfl)

const AdaptiveImplicitVerticalAdvection = AbstractAdvectionScheme{<:Any, <:Any, <:AdaptiveVerticallyImplicitDiscretization}

const AIVA = AdaptiveImplicitVerticalAdvection
const ATD = AbstractTimeDiscretization

@inline time_discretization(scheme::AbstractAdvectionScheme) = scheme.time_discretization
