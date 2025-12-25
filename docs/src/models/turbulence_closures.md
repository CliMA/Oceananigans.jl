# [Turbulence closures](@id turbulence_closures)

A turbulent diffusivity closure representing the effects of viscous dissipation and diffusion can
be passed via the `closure` keyword.

See [turbulence closures](@ref numerical_closures) and [large eddy simulation](@ref numerical_les) for
more details on turbulent diffusivity closures.

## Isotropic diffusivity

To use constant isotropic values for the viscosity ``\nu`` and diffusivity ``\kappa``,
use [`ScalarDiffusivity`](@ref):

```jldoctest
julia> using Oceananigans

julia> closure = ScalarDiffusivity(ν=1e-2, κ=1e-2)
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=0.01, κ=0.01)
```

## Horizontal and vertical diffusivity

To specify constant values for the horizontal and vertical viscosities, ``\nu_h`` and ``\nu_z``,
and horizontal and vertical diffusivities, ``\kappa_h`` and ``\kappa_z``, use
[`HorizontalScalarDiffusivity`](@ref) and [`VerticalScalarDiffusivity`](@ref):

```jldoctest
julia> using Oceananigans

julia> horizontal_closure = HorizontalScalarDiffusivity(ν=1e-3, κ=2e-3)
HorizontalScalarDiffusivity{ExplicitTimeDiscretization}(ν=0.001, κ=0.002)

julia> vertical_closure = VerticalScalarDiffusivity(ν=1e-3, κ=2e-3)
VerticalScalarDiffusivity{ExplicitTimeDiscretization}(ν=0.001, κ=0.002)
```

After that you can set, e.g., `closure = (horizontal_closure, vertical_closure)` when constructing
the model so that all components will be taken into account when calculating the diffusivity term.

## Biharmonic diffusivity

For biharmonic (fourth-order) diffusivity, use [`ScalarBiharmonicDiffusivity`](@ref),
[`HorizontalScalarBiharmonicDiffusivity`](@ref), or [`VerticalScalarBiharmonicDiffusivity`](@ref):

```jldoctest
julia> using Oceananigans

julia> closure = HorizontalScalarBiharmonicDiffusivity(ν=1e8, κ=1e8)
ScalarBiharmonicDiffusivity{HorizontalFormulation}(ν=1.0e8, κ=1.0e8)
```

### Discrete diffusivity functions

A powerful feature is the ability to specify viscosity or diffusivity as a discrete function
of grid indices and model state. This is useful for resolution-dependent coefficients.
For example, to set ``\nu = A^2 / \lambda`` where ``A`` is the grid cell area and ``\lambda``
is a damping timescale:

```julia
using Oceananigans
using Oceananigans.Units

@inline νhb(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, λ) =
    Oceananigans.Operators.Az(i, j, k, grid, ℓx, ℓy, ℓz)^2 / λ

closure = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true, parameters=15days)
```

The `ℓx`, `ℓy`, `ℓz` arguments indicate the grid location (`Center()` or `Face()`) where
the diffusivity is evaluated.

## Tracer-specific diffusivities

You can also set different diffusivities for each tracer in your simulation by passing
a [NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple) as the argument ``\kappa``:

```jldoctest
julia> using Oceananigans

julia> ScalarDiffusivity(ν=1e-6, κ=(S=1e-7, T=1e-10))
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=1.0e-6, κ=(S=1.0e-7, T=1.0e-10))
```

The example above sets a viscosity of `1e-6`, a diffusivity for a tracer called `T` of `1e-7`,
and a diffusivity for a tracer called `S` of `1e-10`. Specifying diffusivities this way is also valid
for [`HorizontalScalarDiffusivity`](@ref), [`VerticalScalarDiffusivity`](@ref), and biharmonic closures.
If this method is used, diffusivities for all tracers need to be specified.

## Smagorinsky closures

The Smagorinsky closure computes an eddy viscosity from the resolved strain rate.
Two variants are available:

- [`SmagorinskyLilly`](@ref): includes a buoyancy-based stability correction (the default)
- [`DynamicSmagorinsky`](@ref): dynamically computes the coefficient from the resolved flow

```jldoctest
julia> using Oceananigans

julia> SmagorinskyLilly()
Smagorinsky closure with
├── coefficient = LillyCoefficient(smagorinsky = 0.16, reduction_factor = 1.0)
└── Pr = 1.0

julia> DynamicSmagorinsky()
DynamicSmagorinsky{Float64}:
├── averaging = Oceananigans.TurbulenceClosures.Smagorinskys.LagrangianAveraging()
├── schedule = IterationInterval(1, 0)
├── Pr = 1.0
└── minimum_numerator = 1.0e-32
```

[`DynamicSmagorinsky`](@ref) supports Lagrangian averaging (the default) or directional averaging:

```jldoctest
julia> using Oceananigans

julia> DynamicSmagorinsky(averaging=(1, 2))  # average in x and y
DynamicSmagorinsky{Float64}:
├── averaging = (1, 2)
├── schedule = IterationInterval(1, 0)
├── Pr = 1.0
└── minimum_numerator = 1.0e-32
```

## Anisotropic minimum dissipation

To use the constant anisotropic minimum dissipation (AMD) LES closure, use
[`AnisotropicMinimumDissipation`](@ref):

```jldoctest
julia> using Oceananigans

julia> closure = AnisotropicMinimumDissipation()
AnisotropicMinimumDissipation{ExplicitTimeDiscretization} turbulence closure with:
           Poincaré constant for momentum eddy viscosity Cν: 0.3333333333333333
    Poincaré constant for tracer(s) eddy diffusivit(ies) Cκ: 0.3333333333333333
                        Buoyancy modification multiplier Cb: nothing
```

No parameters are required although they may be specified. By default, the background viscosity and diffusivity
are assumed to be the molecular values for seawater.

## Convective adjustment vertical diffusivity

To use a convective adjustment scheme that applies enhanced values for vertical diffusivity ``\kappa_z`` and/or
viscosity ``\nu_z``, anytime and anywhere the background stratification becomes unstable,
use [`ConvectiveAdjustmentVerticalDiffusivity`](@ref):

```jldoctest
julia> using Oceananigans

julia> closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, background_κz = 1e-3)
ConvectiveAdjustmentVerticalDiffusivity{VerticallyImplicitTimeDiscretization}(background_κz=0.001 convective_κz=1.0 background_νz=0.0 convective_νz=0.0)
```

## Isopycnal skew-symmetric diffusivity

For mesoscale eddy parameterization using the Gent-McWilliams/Redi scheme, use
[`IsopycnalSkewSymmetricDiffusivity`](@ref). This closure requires import from `TurbulenceClosures`:

```julia
using Oceananigans
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity

closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3)
```

The `κ_skew` parameter controls the eddy-induced (bolus) transport, while `κ_symmetric`
controls along-isopycnal diffusion.

## CATKE vertical diffusivity

[`CATKEVerticalDiffusivity`](@ref) is a TKE-based closure for vertical mixing by small-scale
ocean turbulence. It uses a prognostic equation for turbulent kinetic energy (the `:e` tracer).

!!! note "HydrostaticFreeSurfaceModel only"
    `CATKEVerticalDiffusivity` is currently only supported by `HydrostaticFreeSurfaceModel`.

```julia
using Oceananigans
closure = CATKEVerticalDiffusivity()
```

## TKE-Dissipation vertical diffusivity

[`TKEDissipationVerticalDiffusivity`](@ref) is a two-equation closure (k-ε) for vertical mixing
that uses prognostic equations for both turbulent kinetic energy (`:e`) and its dissipation rate (`:ϵ`).

!!! note "HydrostaticFreeSurfaceModel only"
    `TKEDissipationVerticalDiffusivity` is currently only supported by `HydrostaticFreeSurfaceModel`.

```julia
using Oceananigans
closure = TKEDissipationVerticalDiffusivity()
```
