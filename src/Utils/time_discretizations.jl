"""
    abstract type AbstractTimeDiscretization

Abstract supertype for time-discretizations of advection schemes, turbulence closures, and
boundary conditions.
"""
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

"""
    struct ImplicitExplicitTimeDiscretization <: AbstractTimeDiscretization

An implicit-explicit (IMEX) time-discretization for an affine `Flux` boundary condition
`J(φ_b) = Fₑ + λ φ_b`. The explicit part `Fₑ` is integrated through the tendency like an
ordinary flux boundary condition, while the linear part `λ φ_b` is integrated implicitly by
the vertical tridiagonal solver.
"""
struct ImplicitExplicitTimeDiscretization <: AbstractTimeDiscretization end

Base.summary(::ImplicitExplicitTimeDiscretization) = "ImplicitExplicitTimeDiscretization"
