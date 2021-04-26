# Turbulent diffusivity closures and large eddy simulation models

A turbulent diffusivity closure representing the effects of viscous dissipation and diffusion can be passed via the
`closure` keyword.

See [turbulence closures](@ref numerical_closures) and [large eddy simulation](@ref numerical_les) for more details
on turbulent diffusivity closures.

## Constant isotropic diffusivity

To use constant isotropic values for the viscosity ν and diffusivity κ you can use [`IsotropicDiffusivity`](@ref)

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> closure = IsotropicDiffusivity(ν=1e-2, κ=1e-2)
IsotropicDiffusivity: ν=0.01, κ=0.01
```

## Constant anisotropic diffusivity

To specify constant values for the horizontal and vertical viscosities, ``\nu_h`` and ``\nu_v``, and horizontal and vertical
diffusivities, ``\kappa_h`` and ``\kappa_v``, you can use [`AnisotropicDiffusivity`](@ref)

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> closure = AnisotropicDiffusivity(νh=1e-3, νz=5e-2, κh=2e-3, κz=1e-1)
AnisotropicDiffusivity: (νx=0.001, νy=0.001, νz=0.05), (κx=0.002, κy=0.002, κz=0.1)
```

## Smagorinsky-Lilly

To use the Smagorinsky-Lilly LES closure, no parameters are required

```jldoctest
julia> using Oceananigans.TurbulenceClosures

julia> closure = SmagorinskyLilly()
SmagorinskyLilly: C=0.23, Cb=1.0, Pr=1.0, ν=1.05e-6, κ=1.46e-7
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
                Background diffusivit(ies) for tracer(s), κ: 1.46e-7
             Background kinematic viscosity for momentum, ν: 1.05e-6
```

no parameters are required although they may be specified. By default, the background viscosity and diffusivity
are assumed to be the molecular values for seawater. For more details see [`AnisotropicMinimumDissipation`](@ref).
