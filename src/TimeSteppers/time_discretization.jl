"""
    abstract type AbstractTimeDiscretization

Abstract supertype for time-discretizations of advection schemes and turbulence closures.
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

struct AdaptiveVerticallyImplicitDiscretization{FT, MC, IF, TI, BI, R} <: AbstractTimeDiscretization
    maximum_explicit_cfl       :: MC
    implicit_fraction          :: IF
    sample_top_levels          :: TI
    sample_bottom_levels       :: BI
    cfl                        :: R
    Δt                         :: R
    realized_implicit_fraction :: R
    median_cfl                 :: R
    max_cfl                    :: R
end

"""
    AdaptiveVerticallyImplicitDiscretization([FT = Oceananigans.defaults.FloatType];
                                             maximum_explicit_cfl = nothing,
                                             implicit_fraction = nothing,
                                             cfl = nothing,
                                             sample_top_levels = 10,
                                             sample_bottom_levels = 10)

An adaptively implicit vertical discretization scheme following Shchepetkin (2015) / CROCO.

Splits vertical advection into explicit and implicit parts based on the local vertical Courant number
`α = |w| Δt / Δz`. When `α ≤ CFL_threshold`, advection is fully explicit using `explicit_scheme`. When
`α > CFL_threshold`, the vertical velocity is decomposed as `w = wᵉ + wⁱ` where `wᵉ` is CFL-limited and
`wⁱ` is treated with implicit first-order upwind in the existing tridiagonal solver.

`CFL_threshold` is selected by exactly one control policy:

- `maximum_explicit_cfl`: use a fixed explicit CFL threshold everywhere.
- `implicit_fraction`: compute the threshold dynamically so approximately this fraction of sampled cells
  use an implicit remainder. The sampled population includes the top `sample_top_levels` wet cells and
  the bottom-adjacent `sample_bottom_levels` wet cells in each water column, with overlap removed in
  shallow columns.

The splitting function is:

    f(α, CFL_threshold) = max(1, α / CFL_threshold)
    wᵉ = w / f  (explicit, CFL-limited)
    wⁱ = w - wᵉ (implicit, first-order upwind)

Keyword Arguments
=================

- `maximum_explicit_cfl`: Maximum vertical CFL for the explicit part.
- `implicit_fraction`: Fraction of active cells allowed to use an implicit remainder.
- `cfl`: Backwards-compatible alias for `maximum_explicit_cfl`.
- `sample_top_levels`: Number of top wet cells sampled per column when `implicit_fraction` is used.
- `sample_bottom_levels`: Number of bottom-adjacent wet cells sampled per column when `implicit_fraction` is used.

!!! note
    Exactly one of `maximum_explicit_cfl`, `implicit_fraction`, or `cfl` must be provided.
    `implicit_fraction = 0` keeps all active cells explicit, while `implicit_fraction = 1`
    makes the threshold the minimum active-cell CFL and thus sends nearly all cells to the
    implicit remainder, up to ties at the minimum.

Example
=======

```jldoctest
julia> using Oceananigans

julia> AdaptiveVerticallyImplicitDiscretization(maximum_explicit_cfl=0.3)
AdaptiveVerticallyImplicitDiscretization:
├── control: maximum_explicit_cfl
├── value: 0.3
└── resolved_cfl: 0.3
```
"""
function AdaptiveVerticallyImplicitDiscretization(FT::DataType = Oceananigans.defaults.FloatType;
                                                  maximum_explicit_cfl = nothing,
                                                  implicit_fraction = nothing,
                                                  cfl = nothing,
                                                  sample_top_levels = 10,
                                                  sample_bottom_levels = 10)
    !isnothing(cfl) && !isnothing(maximum_explicit_cfl) &&
        throw(ArgumentError("Specify only one of `cfl` or `maximum_explicit_cfl`."))

    maximum_explicit_cfl = isnothing(maximum_explicit_cfl) ? cfl : maximum_explicit_cfl

    provided_modes = (!isnothing(maximum_explicit_cfl)) + (!isnothing(implicit_fraction))
    provided_modes == 1 ||
        throw(ArgumentError("Exactly one of `maximum_explicit_cfl` or `implicit_fraction` must be specified."))

    if !isnothing(maximum_explicit_cfl)
        maximum_explicit_cfl = convert(FT, maximum_explicit_cfl)
        maximum_explicit_cfl >= zero(FT) ||
            throw(ArgumentError("`maximum_explicit_cfl` must be non-negative."))
    end

    if !isnothing(implicit_fraction)
        implicit_fraction = convert(FT, implicit_fraction)
        zero(FT) <= implicit_fraction <= one(FT) ||
            throw(ArgumentError("`implicit_fraction` must lie in the interval [0, 1]."))
    end

    sample_top_levels = convert(Int, sample_top_levels)
    sample_bottom_levels = convert(Int, sample_bottom_levels)
    sample_top_levels >= 0 || throw(ArgumentError("`sample_top_levels` must be non-negative."))
    sample_bottom_levels >= 0 || throw(ArgumentError("`sample_bottom_levels` must be non-negative."))

    resolved_cfl = Ref(isnothing(maximum_explicit_cfl) ? zero(FT) : maximum_explicit_cfl)
    Δt = Ref(zero(FT))
    realized_implicit_fraction = Ref(zero(FT))
    median_cfl = Ref(zero(FT))
    max_cfl = Ref(zero(FT))

    return AdaptiveVerticallyImplicitDiscretization{FT,
                                                    typeof(maximum_explicit_cfl),
                                                    typeof(implicit_fraction),
                                                    typeof(sample_top_levels),
                                                    typeof(sample_bottom_levels),
                                                    typeof(resolved_cfl)}(maximum_explicit_cfl,
                                                                         implicit_fraction,
                                                                         sample_top_levels,
                                                                         sample_bottom_levels,
                                                                         resolved_cfl,
                                                                         Δt,
                                                                         realized_implicit_fraction,
                                                                         median_cfl,
                                                                         max_cfl)
end

Adapt.adapt_structure(to, a::AdaptiveVerticallyImplicitDiscretization) =
    AdaptiveVerticallyImplicitDiscretization{typeof(a.Δt[]),
                                             typeof(a.maximum_explicit_cfl),
                                             typeof(a.implicit_fraction),
                                             typeof(a.sample_top_levels),
                                             typeof(a.sample_bottom_levels),
                                             typeof(Adapt.adapt(to, a.cfl))}(a.maximum_explicit_cfl,
                                                                             a.implicit_fraction,
                                                                             a.sample_top_levels,
                                                                             a.sample_bottom_levels,
                                                                             Adapt.adapt(to, a.cfl),
                                                                             Adapt.adapt(to, a.Δt),
                                                                             Adapt.adapt(to, a.realized_implicit_fraction),
                                                                             Adapt.adapt(to, a.median_cfl),
                                                                             Adapt.adapt(to, a.max_cfl))

@inline unwrap_time_discretization_property(x) = x
@inline unwrap_time_discretization_property(x::Base.RefValue) = x[]

function adaptive_implicit_vertical_advection_diagnostics(td::AdaptiveVerticallyImplicitDiscretization)
    return (; resolved_cfl = unwrap_time_discretization_property(td.cfl),
             realized_implicit_fraction = unwrap_time_discretization_property(td.realized_implicit_fraction),
             median_cfl = unwrap_time_discretization_property(td.median_cfl),
             max_cfl = unwrap_time_discretization_property(td.max_cfl))
end

adaptive_implicit_vertical_advection_diagnostics(scheme) =
    adaptive_implicit_vertical_advection_diagnostics(time_discretization(scheme))

adaptive_implicit_vertical_advection_control(a::AdaptiveVerticallyImplicitDiscretization) =
    isnothing(a.maximum_explicit_cfl) ? :implicit_fraction : :maximum_explicit_cfl

adaptive_implicit_vertical_advection_control_value(a::AdaptiveVerticallyImplicitDiscretization) =
    isnothing(a.maximum_explicit_cfl) ? a.implicit_fraction : a.maximum_explicit_cfl

Base.summary(a::AdaptiveVerticallyImplicitDiscretization) =
    string("AdaptiveVerticallyImplicitDiscretization(",
           adaptive_implicit_vertical_advection_control(a), "=",
           adaptive_implicit_vertical_advection_control_value(a), ")")

function Base.show(io::IO, a::AdaptiveVerticallyImplicitDiscretization)
    diagnostics = adaptive_implicit_vertical_advection_diagnostics(a)
    print(io, "AdaptiveVerticallyImplicitDiscretization:", "\n",
              "├── control: ", adaptive_implicit_vertical_advection_control(a), "\n",
              "├── value: ", adaptive_implicit_vertical_advection_control_value(a), "\n",
              "└── resolved_cfl: ", diagnostics.resolved_cfl)
end

"""
    time_discretization(scheme_or_closure)

Return the time-discretization associated with an advection scheme or turbulence closure.
Extended in `Advection` and `TurbulenceClosures` modules.
"""
function time_discretization end
