# [Turbulence closures, viscosities, and diffusivities](@id turbulence_closures)

When a turbulence closure is added to the constructor of either [`NonhydrostaticModel`](@ref) or
[`HydrostaticFreeSurfaceModel`](@ref) via the keyword argument `closure`, it contributes a diffusive
flux divergence to the tendency of momentum and tracer equations, and typically represents the
effects of turbulent processes that are not explicitly resolved. Turbulence closures can also
represent molecular viscosity and diffusion of heat, salt, or other tracers.

Below we detail available options for turbulence closures. A `tuple` of closures can be used to
represent multiple closures added together.

### Scalar diffusivity

To use constant isotropic values for the viscosity ``\nu`` and diffusivity ``\kappa``,
use [`ScalarDiffusivity`](@ref):

```jldoctest
julia> using Oceananigans

julia> closure = ScalarDiffusivity(ν=1e-2, κ=1e-2)
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=0.01, κ=0.01)
```

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

#### Tracer-specific diffusivities

Different diffusivities for each tracer can be set by passing a `NamedTuple` for ``\kappa``:

```jldoctest
julia> using Oceananigans

julia> ScalarDiffusivity(ν=1e-6, κ=(S=1e-7, T=1e-10))
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=1.0e-6, κ=(S=1.0e-7, T=1.0e-10))
```

The above sets a viscosity of `1e-6`, and diffusivities for tracers `T` and `S` of `1e-7`
and `1e-10` respectively. This method is valid for all scalar diffusivity closures.
This pattern of `NamedTuple` for tracer-specific values is used in other closures as well.

### Biharmonic diffusivity

For biharmonic (fourth-order) diffusivity, use [`ScalarBiharmonicDiffusivity`](@ref),
[`HorizontalScalarBiharmonicDiffusivity`](@ref), or [`VerticalScalarBiharmonicDiffusivity`](@ref):

```jldoctest
julia> using Oceananigans

julia> closure = HorizontalScalarBiharmonicDiffusivity(ν=1e8, κ=1e8)
ScalarBiharmonicDiffusivity{HorizontalFormulation}(ν=1.0e8, κ=1.0e8)
```

#### Discrete diffusivity functions

Viscosity or diffusivity can be specified as a discrete function of grid indices and model state.
This is useful for resolution-dependent coefficients.
For example, we can set ``\nu = A^2 / \lambda``, where ``A`` is the grid cell area and ``\lambda``
is a damping timescale:

```jldoctest closures
julia> using Oceananigans, Oceananigans.Units

julia> @inline νhb(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, λ) =
           Oceananigans.Operators.Az(i, j, k, grid, ℓx, ℓy, ℓz)^2 / λ
νhb (generic function with 1 method)

julia> closure = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true, parameters=15days)
ScalarBiharmonicDiffusivity{HorizontalFormulation}(ν=Oceananigans.TurbulenceClosures.DiscreteDiffusionFunction{Nothing, Nothing, Nothing, Float64, typeof(νhb)}, κ=0.0)
```

The `ℓx`, `ℓy`, `ℓz` arguments indicate the grid location (`Center()` or `Face()`) where
the diffusivity is evaluated.

### Tupled closures

Multiple closures can be combined in a tuple. All closure contributions are summed:

```jldoctest
julia> using Oceananigans

julia> closure = (HorizontalScalarDiffusivity(ν=1e-3), VerticalScalarDiffusivity(ν=1e-4))
(HorizontalScalarDiffusivity{ExplicitTimeDiscretization}(ν=0.001, κ=0.0), VerticalScalarDiffusivity{ExplicitTimeDiscretization}(ν=0.0001, κ=0.0))
```

See the [hydrostatic modeling example](@ref hydrostatic-example) below for a more complex tuple.

## Closures for large eddy simulation

These closures compute eddy viscosity and diffusivity from resolved flow features
for large eddy simulation (LES).

### `SmagorinskyLilly`

[`SmagorinskyLilly`](@ref) computes an eddy viscosity from the resolved strain rate
with a buoyancy-based stability correction, following the classic work of
[Smagorinsky1958](@citet), [Smagorinsky63](@citet), [Lilly62](@citet), and [Lilly66](@citet):

```jldoctest
julia> using Oceananigans

julia> SmagorinskyLilly()
Smagorinsky closure with
├── coefficient = LillyCoefficient(smagorinsky = 0.16, reduction_factor = 1.0)
└── Pr = 1.0
```

### `DynamicSmagorinsky`

[`DynamicSmagorinsky`](@ref) dynamically computes the Smagorinsky coefficient from the resolved flow,
using the scale-invariant procedure described by [BouZeid05](@citet).
Two averaging methods are available:

**Lagrangian averaging** (default): averages along fluid parcel trajectories:

```jldoctest
julia> using Oceananigans

julia> DynamicSmagorinsky()
DynamicSmagorinsky{Float64}:
├── averaging = Oceananigans.TurbulenceClosures.Smagorinskys.LagrangianAveraging()
├── schedule = IterationInterval(1, 0)
├── Pr = 1.0
└── minimum_numerator = 1.0e-32
```

**Directional averaging**: averages along specified dimensions, useful for flows with
homogeneous directions:

```jldoctest
julia> using Oceananigans

julia> DynamicSmagorinsky(averaging=(1, 2))  # average in x and y
DynamicSmagorinsky{Float64}:
├── averaging = (1, 2)
├── schedule = IterationInterval(1, 0)
├── Pr = 1.0
└── minimum_numerator = 1.0e-32
```

### `AnisotropicMinimumDissipation`

[`AnisotropicMinimumDissipation`](@ref) is an LES closure based on the minimum dissipation principle,
as developed by [Verstappen14](@citet), [Rozema15](@citet), [Abkar16](@citet), and [Verstappen18](@citet):

```jldoctest
julia> using Oceananigans

julia> AnisotropicMinimumDissipation()
AnisotropicMinimumDissipation{ExplicitTimeDiscretization} turbulence closure with:
           Poincaré constant for momentum eddy viscosity Cν: 0.3333333333333333
    Poincaré constant for tracer(s) eddy diffusivit(ies) Cκ: 0.3333333333333333
                        Buoyancy modification multiplier Cb: nothing
```

## Closures for hydrostatic modeling

These closures are designed for hydrostatic models, parameterizing vertical mixing
and mesoscale eddies.

### `ConvectiveAdjustmentVerticalDiffusivity`

[`ConvectiveAdjustmentVerticalDiffusivity`](@ref) applies enhanced vertical diffusivity
whenever and wherever the stratification becomes unstable:

```jldoctest
julia> using Oceananigans

julia> ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, background_κz = 1e-3)
ConvectiveAdjustmentVerticalDiffusivity{VerticallyImplicitTimeDiscretization}(background_κz=0.001 convective_κz=1.0 background_νz=0.0 convective_νz=0.0)
```

### `CATKEVerticalDiffusivity`

[`CATKEVerticalDiffusivity`](@ref) is a TKE-based closure for vertical mixing by small-scale
ocean turbulence, as described by [Wagner25catke](@citet).
It uses a prognostic equation for turbulent kinetic energy (the `:e` tracer).

!!! note "HydrostaticFreeSurfaceModel only"
    `CATKEVerticalDiffusivity` is currently only supported by `HydrostaticFreeSurfaceModel`.

```jldoctest
julia> using Oceananigans

julia> CATKEVerticalDiffusivity()
CATKEVerticalDiffusivity{VerticallyImplicitTimeDiscretization}
├── maximum_tracer_diffusivity: Inf
├── maximum_tke_diffusivity: Inf
├── maximum_viscosity: Inf
├── minimum_tke: 1.0e-9
├── negative_tke_time_scale: 60.0
├── minimum_convective_buoyancy_flux: 1.0e-11
├── tke_time_step: Nothing
├── mixing_length: TKEBasedVerticalDiffusivities.CATKEMixingLength
│   ├── Cˢ:   1.131
│   ├── Cᵇ:   0.28
│   ├── Cʰⁱu: 0.242
│   ├── Cʰⁱc: 0.098
│   ├── Cʰⁱe: 0.548
│   ├── Cˡᵒu: 0.361
│   ├── Cˡᵒc: 0.369
│   ├── Cˡᵒe: 7.863
│   ├── Cᵘⁿu: 0.37
│   ├── Cᵘⁿc: 0.572
│   ├── Cᵘⁿe: 1.447
│   ├── Cᶜu:  3.705
│   ├── Cᶜc:  4.793
│   ├── Cᶜe:  3.642
│   ├── Cᵉc:  0.112
│   ├── Cᵉe:  0.0
│   ├── Cˢᵖ:  0.505
│   ├── CRiᵟ: 1.02
│   └── CRi⁰: 0.254
└── turbulent_kinetic_energy_equation: TKEBasedVerticalDiffusivities.CATKEEquation
    ├── CʰⁱD: 0.579
    ├── CˡᵒD: 1.604
    ├── CᵘⁿD: 0.923
    ├── CᶜD:  3.254
    ├── CᵉD:  0.0
    ├── Cᵂu★: 3.179
    ├── CᵂwΔ: 0.383
    └── Cᵂϵ:  1.0
```

### `TKEDissipationVerticalDiffusivity`

[`TKEDissipationVerticalDiffusivity`](@ref) is a two-equation closure (k-ε) for vertical mixing
that uses prognostic equations for both turbulent kinetic energy (`:e`) and its dissipation rate (`:ϵ`).
For more information about k-ε closures, see [burchard2001comparative](@citet),
[umlauf2003generic](@citet), and [umlauf2005second](@citet).

!!! note "HydrostaticFreeSurfaceModel only"
    `TKEDissipationVerticalDiffusivity` is currently only supported by `HydrostaticFreeSurfaceModel`.

```jldoctest
julia> using Oceananigans

julia> TKEDissipationVerticalDiffusivity()
TKEDissipationVerticalDiffusivity{VerticallyImplicitTimeDiscretization}
├── maximum_tracer_diffusivity: Inf
├── maximum_tke_diffusivity: Inf
├── maximum_dissipation_diffusivity: Inf
├── maximum_viscosity: Inf
├── minimum_tke: 1.0e-6
├── negative_tke_damping_time_scale: 60.0
├── tke_dissipation_time_step: Nothing
├── tke_dissipation_equations: Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities.TKEDissipationEquations{Float64}
│   ├── Cᵋϵ: 1.92
│   ├── Cᴾϵ: 1.44
│   ├── Cᵇϵ⁺: -0.65
│   ├── Cᵇϵ⁻: -0.65
│   ├── Cᵂu★: 0.0
│   └── CᵂwΔ: 0.0
└── stability_functions: VariableStabilityFunctions{Float64}:
    ├── Cσe: 1.0
    ├── Cσϵ: 1.2
    ├── Cu₀: 0.1067
    ├── Cu₁: 0.0173
    ├── Cu₂: -0.0001205
    ├── Cc₀: 0.112
    ├── Cc₁: 0.003766
    ├── Cc₂: 0.0008871
    ├── Cd₀: 1.0
    ├── Cd₁: 0.2398
    ├── Cd₂: 0.02872
    ├── Cd₃: 0.005154
    ├── Cd₄: 0.00693
    └── Cd₅: -0.0003372    
```

### `IsopycnalSkewSymmetricDiffusivity`

[`IsopycnalSkewSymmetricDiffusivity`](@ref) parameterizes mesoscale eddies using the
Gent-McWilliams/Redi scheme, as developed by [GentMcWilliams90](@citet) and [Redi82](@citet).
This closure requires to be imported from `TurbulenceClosures`:

```jldoctest
julia> using Oceananigans

julia> using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity

julia> IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3)
IsopycnalSkewSymmetricDiffusivity:
├── κ_skew: 1000.0
├── κ_symmetric: 1000.0
├── isopycnal_tensor: Oceananigans.TurbulenceClosures.SmallSlopeIsopycnalTensor{Float64}
└── slope_limiter: Oceananigans.TurbulenceClosures.FluxTapering{Float64}
```

The `κ_skew` parameter controls the eddy-induced (bolus) transport, while `κ_symmetric`
controls along-isopycnal diffusion.

### [Combining closures for hydrostatic simulations](@id hydrostatic-example)

A typical hydrostatic simulation might combine resolution-dependent horizontal biharmonic
viscosity with a TKE-based vertical mixing scheme:

```jldoctest closures
julia> horizontal_closure = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true, parameters=5days)
ScalarBiharmonicDiffusivity{HorizontalFormulation}(ν=Oceananigans.TurbulenceClosures.DiscreteDiffusionFunction{Nothing, Nothing, Nothing, Float64, typeof(νhb)}, κ=0.0)

julia> vertical_closure = CATKEVerticalDiffusivity();

julia> closure = (horizontal_closure, vertical_closure);

julia> summary(horizontal_closure)
"ScalarBiharmonicDiffusivity{HorizontalFormulation}(ν=Oceananigans.TurbulenceClosures.DiscreteDiffusionFunction{Nothing, Nothing, Nothing, Float64, typeof(νhb)}, κ=0.0)"
```
