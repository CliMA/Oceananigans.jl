# Turbulent diffusivity closures and large eddy simulation models
A turbulent diffusivty closure representing the effects of viscous dissipation and diffusion can be passed via the
`closure` keyword.

See [turbulence closures](@ref numerical_closures) and [large eddy simulation](@ref numerical_les) for more details
on turbulent diffusivity closures.

## Constant isotropic diffusivity
To use constant isotropic values for the viscosity ν and diffusivity κ you can use `ConstantIsotropicDiffusivity`
```@example
closure = ConstantIsotropicDiffusivity(ν=1e-2, κ=1e-2)
```
## Constant anisotropic diffusivity
To specify constant values for the horizontal and vertical viscosities, $\nu_h$ and $\nu_v$, and horizontal and vertical
diffusivities, $\kappa_h$ and $\kappa_v$, you can use `ConstantAnisotropicDiffusivity`
```@example
closure = ConstantAnisotropicDiffusivity(νh=1e-3, νv=5e-2, κh=2e-3, κv=1e-1)
```

## Smagorinsky-Lilly
To use the Smagorinsky-Lilly LES closure, no parameters are required
```@example
closure = SmagorinskyLilly()
```
although they may be specified. By default, the background viscosity and diffusivity are assumed to be the molecular
values for seawater. For more details see [`SmagorinskyLilly`](@ref).

## Anisotropic minimum dissipation
To use the constant anisotropic minimum dissipation (AMD) LES closure, no parameters are required
```@example
closure = AnisotropicMinimumDissipation()
```
although they may be specified. By default, the background viscosity and diffusivity are assumed to be the molecular
values for seawater. For more details see [`AnisotropicMinimumDissipation`](@ref).
