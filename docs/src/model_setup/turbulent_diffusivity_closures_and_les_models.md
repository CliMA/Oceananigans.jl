# Turbulent diffusivity closures and large eddy simulation models

A turbulent diffusivity closure representing the effects of viscous dissipation and diffusion can
be passed via the `closure` keyword.

See [turbulence closures](@ref numerical_closures) and [large eddy simulation](@ref numerical_les) for
more details on turbulent diffusivity closures.

## Constant isotropic diffusivity

To use constant isotropic values for the viscosity ``\nu`` and diffusivity ``\kappa`` you can use `ScalarDiffusivity`:

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> closure = ScalarDiffusivity(ν=1e-2, κ=1e-2)
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=0.01, κ=0.01)
```

## Constant anisotropic diffusivity

To specify constant values for the horizontal and vertical viscosities, ``\nu_h`` and ``\nu_z``,
and horizontal and vertical diffusivities, ``\kappa_h`` and ``\kappa_z``, you can use
`HorizontalScalarDiffusivity()` and `VerticalScalarDiffusivity()`, e.g.,

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> horizontal_closure = HorizontalScalarDiffusivity(ν=1e-3, κ=2e-3)
HorizontalScalarDiffusivity{ExplicitTimeDiscretization}(ν=0.001, κ=0.002)

julia> vertical_closure = VerticalScalarDiffusivity(ν=1e-3, κ=2e-3)
VerticalScalarDiffusivity{ExplicitTimeDiscretization}(ν=0.001, κ=0.002)
```

After that you can set, e.g., `closure = (horizontal_closure, vertical_closure)` when constructing
the model so that all components will be taken into account when calculating the diffusivity term.
Note that `VerticalScalarDiffusivity` and `HorizontalScalarDiffusivity` are implemented using [different
schemes](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#horizontal-dissipation)
with different conservation properties.

## Tracer-specific diffusivities

You can also set different diffusivities for each tracer in your simulation by passing
a [NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple) as the argument ``\kappa``:

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> ScalarDiffusivity(ν=1e-6, κ=(S=1e-7, T=1e-10))
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=1.0e-6, κ=(S=1.0e-7, T=1.0e-10))
```

The example above sets a viscosity of `1e-6`, a diffusivity for a tracer called `T` of `1e-7`,
and a diffusivity for a tracer called `S` of `1e-10`. Specifying diffusivities this way is also valid
for `HorizontalScalarDiffusivity` and `VerticalScalarDiffusivity`. If this method is used, diffusivities
for all tracers need to be specified.


## Smagorinsky-Lilly

To use the Smagorinsky-Lilly LES closure, no parameters are required

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> closure = SmagorinskyLilly()
SmagorinskyLilly: C=0.16, Cb=1.0, Pr=1.0
```

although they may be specified. By default, the background viscosity and diffusivity are assumed to
be the molecular values for seawater. For more details see [`SmagorinskyLilly`](@ref).

## Anisotropic minimum dissipation

To use the constant anisotropic minimum dissipation (AMD) LES closure,

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> closure = AnisotropicMinimumDissipation()
AnisotropicMinimumDissipation{ExplicitTimeDiscretization} turbulence closure with:
           Poincaré constant for momentum eddy viscosity Cν: 0.08333333333333333
    Poincaré constant for tracer(s) eddy diffusivit(ies) Cκ: 0.08333333333333333
                        Buoyancy modification multiplier Cb: nothing
```

no parameters are required although they may be specified. By default, the background viscosity and diffusivity
are assumed to be the molecular values for seawater. For more details see [`AnisotropicMinimumDissipation`](@ref).

## Convective Adjustment Vertical Diffusivity--Viscosity

To use the a convective adjustment scheme that applies enhanced values for vertical diffusivity ``\kappa_z`` and/or
viscosity ``\nu_z``, anytime and anywhere the background stratification becomes unstable.

```jldoctest
julia> using Oceananigans

julia> closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, background_κz = 1e-3)
ConvectiveAdjustmentVerticalDiffusivity{VerticallyImplicitTimeDiscretization}(background_κz=0.001 convective_κz=1.0 background_νz=0.0 convective_νz=0.0)
```
