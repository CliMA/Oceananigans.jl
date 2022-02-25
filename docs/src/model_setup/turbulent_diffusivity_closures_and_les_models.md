# Turbulent diffusivity closures and large eddy simulation models

A turbulent diffusivity closure representing the effects of viscous dissipation and diffusion can be passed via the
`closure` keyword.

See [turbulence closures](@ref numerical_closures) and [large eddy simulation](@ref numerical_les) for more details
on turbulent diffusivity closures.

## Constant isotropic diffusivity

To use constant isotropic values for the viscosity ``\nu`` and diffusivity ``\kappa`` you can use `ScalarDiffusivity`

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> closure = ScalarDiffusivity(ν=1e-2, κ=1e-2)
ScalarDiffusivity{ExplicitTimeDiscretization, ThreeDimensional}(ν=0.01, κ=0.01)
```

## Constant anisotropic diffusivity

To specify constant values for the horizontal and vertical viscosities, ``\nu_h`` and ``\nu_z``, and horizontal and vertical
diffusivities, ``\kappa_h`` and ``\kappa_z``, you can use `HorizontalScalarDiffusivity()` and
`VerticalScalarDiffusivity()`

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> horizontal_closure = HorizontalScalarDiffusivity(ν=1e-3, κ=2e-3)
ScalarDiffusivity{ExplicitTimeDiscretization, HorizontalFormulation}(ν=0.001, κ=0.002)

julia> vertical_closure = VerticalScalarDiffusivity(ν=1e-3, κ=2e-3)
ScalarDiffusivity{ExplicitTimeDiscretization, VerticalFormulation}(ν=0.001, κ=0.002)
```

After that you can set `closure = (horizontal_closure, vertical_closure)` when constructing the
model so that all components will be taken into account when calculating the diffusivity term. Note
that `VerticalScalarDiffusivity` and `HorizontalScalarDiffusivity` are implemented using [different
schemes](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-dissipation)
with different conservation properties.

## Smagorinsky-Lilly

To use the Smagorinsky-Lilly LES closure, no parameters are required

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> closure = SmagorinskyLilly()
SmagorinskyLilly: C=0.16, Cb=1.0, Pr=1.0, ν=0.0, κ=0.0
```

although they may be specified. By default, the background viscosity and diffusivity are assumed to be the molecular
values for seawater. For more details see [`SmagorinskyLilly`](@ref).

## Anisotropic minimum dissipation

To use the constant anisotropic minimum dissipation (AMD) LES closure,

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> closure = AnisotropicMinimumDissipation()
AnisotropicMinimumDissipation{Float64} turbulence closure with:
           Poincaré constant for momentum eddy viscosity Cν: 0.08333333333333333
    Poincaré constant for tracer(s) eddy diffusivit(ies) Cκ: 0.08333333333333333
                        Buoyancy modification multiplier Cb: nothing
                Background diffusivit(ies) for tracer(s), κ: 0.0
             Background kinematic viscosity for momentum, ν: 0.0
```

no parameters are required although they may be specified. By default, the background viscosity and diffusivity
are assumed to be the molecular values for seawater. For more details see [`AnisotropicMinimumDissipation`](@ref).

## Convective Adjustment Vertical Diffusivity--Viscosity

To use the a convective adjustment scheme that applies enhanced values for vertical diffusivity ``\kappa_z`` and/or
viscosity ``\nu_z``, anytime and anywhere the background stratification becomes unstable.

```jldoctest
julia> using Oceananigans

julia> closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, background_κz = 1e-3)
ConvectiveAdjustmentVerticalDiffusivity: (background_κz=0.001, convective_κz=1.0, background_νz=0.0, convective_νz=0.0)
```
