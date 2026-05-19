abstract type AbstractTimeDiscretization end

"""
    struct ExplicitTimeDiscretization <: AbstractTimeDiscretization

A fully-explicit time-discretization of a `TurbulenceClosure`.
"""
struct ExplicitTimeDiscretization <: AbstractTimeDiscretization end

Base.summary(::ExplicitTimeDiscretization) = "ExplicitTimeDiscretization"

"""
    struct VerticallyImplicitTimeDiscretization <: AbstractTimeDiscretization

A vertically-implicit time-discretization of a `TurbulenceClosure`.

This implies that a flux divergence such as ``𝛁 ⋅ 𝐪`` at the ``n``-th timestep is
time-discretized as

```julia
[∇ ⋅ q]ⁿ = [explicit_flux_divergence]ⁿ + [∂z (κ ∂z c)]ⁿ⁺¹
```
"""
struct VerticallyImplicitTimeDiscretization <: AbstractTimeDiscretization end

Base.summary(::VerticallyImplicitTimeDiscretization) = "VerticallyImplicitTimeDiscretization"

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

- `cfl`: Maximum vertical CFL for the explicit part (default: `0.9`).
"""
struct AdaptiveVerticallyImplicitDiscretization{FT, R} <: AbstractTimeDiscretization
    cfl :: FT
    Δt  :: R 
end

function AdaptiveVerticallyImplicitDiscretization(FT::DataType = Oceananigans.defaults.FloatType; cfl = 0.5)
    cfl = convert(FT, cfl)
    Δt  = Ref(zero(FT))
    return AdaptiveVerticallyImplicitDiscretization(explicit_scheme, cfl, Δt)
end

Adapt.adapt_structure(to, a::AdaptiveVerticallyImplicitDiscretization) =
    AdaptiveVerticallyImplicitDiscretization(a.cfl, a.Δt)

Base.summary(a::AdaptiveVerticallyImplicitDiscretization) =
    string("AdaptiveVerticallyImplicitDiscretization(cfl=$(a.cfl))")

Base.show(io::IO, a::AdaptiveVerticallyImplicitDiscretization) =
    print(io, "AdaptiveVerticallyImplicitDiscretization:", "\n",
              "└── cfl: ", a.cfl)

const AdaptiveImplicitVerticalAdvection = AbstractAdvectionScheme{<:Any, <:Any, <:AdaptiveVerticallyImplicitDiscretization}

const AIVA = AdaptiveImplicitVerticalAdvection
const ATD = AbstractTimeDiscretization

@inline advective_tracer_flux_x(i, j, k, grid, ::ATD, args...) = advective_tracer_flux_x(i, j, k, grid, args...)
@inline advective_tracer_flux_y(i, j, k, grid, ::ATD, args...) = advective_tracer_flux_y(i, j, k, grid, args...)
@inline advective_tracer_flux_z(i, j, k, grid, ::ATD, args...) = advective_tracer_flux_z(i, j, k, grid, args...)

@inline advective_momentum_flux_Uu(i, j, k, grid, ::ATD, args...) = advective_momentum_flux_Uu(i, j, k, grid, args...)
@inline advective_momentum_flux_Vu(i, j, k, grid, ::ATD, args...) = advective_momentum_flux_Vu(i, j, k, grid, args...)
@inline advective_momentum_flux_Wu(i, j, k, grid, ::ATD, args...) = advective_momentum_flux_Wu(i, j, k, grid, args...)

@inline advective_momentum_flux_Uv(i, j, k, grid, ::ATD, args...) = advective_momentum_flux_Uv(i, j, k, grid, args...)
@inline advective_momentum_flux_Vv(i, j, k, grid, ::ATD, args...) = advective_momentum_flux_Vv(i, j, k, grid, args...)
@inline advective_momentum_flux_Wv(i, j, k, grid, ::ATD, args...) = advective_momentum_flux_Wv(i, j, k, grid, args...)

@inline advective_momentum_flux_Uw(i, j, k, grid, ::ATD, args...) = advective_momentum_flux_Uw(i, j, k, grid, args...)
@inline advective_momentum_flux_Vw(i, j, k, grid, ::ATD, args...) = advective_momentum_flux_Vw(i, j, k, grid, args...)
@inline advective_momentum_flux_Ww(i, j, k, grid, ::ATD, args...) = advective_momentum_flux_Ww(i, j, k, grid, args...)

@inline time_discretization(scheme::AbstractAdvectionScheme) = scheme.vertical_discretization