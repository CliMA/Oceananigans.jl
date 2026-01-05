# Stokes drift

Stokes drift represents the net Lagrangian transport of fluid particles due to surface gravity waves.
When waves pass through a fluid, particles do not simply oscillate back and forth; instead, they undergo
a slow net drift in the direction of wave propagation. This effect is important for upper ocean dynamics,
particularly for processes like Langmuir turbulence.

!!! info "Model compatibility"
    Currently, only [`NonhydrostaticModel`](@ref) supports Stokes drift. The effects of surface waves
    are modeled via the Craik-Leibovich approximation; see the
    [physics documentation](@ref surface_gravity_waves) for more information.

Oceananigans provides two types for specifying Stokes drift:

- [`UniformStokesDrift`](@ref): For Stokes drift that varies only in the vertical direction and time.
- [`StokesDrift`](@ref): For fully three-dimensional, spatially-varying Stokes drift fields.

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

## `UniformStokesDrift`

[`UniformStokesDrift`](@ref) is used when the surface wave field is horizontally uniform, such that
the Stokes drift velocity varies only with depth ``z`` and time ``t``:

```math
\boldsymbol{u}^S = u^S(z, t) \hat{\boldsymbol{x}} + v^S(z, t) \hat{\boldsymbol{y}}
```

This is the typical choice for idealized simulations of Langmuir turbulence, where waves propagate
uniformly across the domain.

To construct a [`UniformStokesDrift`](@ref), you must provide functions for the vertical derivatives
of the Stokes drift velocity components. The required derivatives are `∂z_uˢ` and `∂z_vˢ`. If the
Stokes drift is time-dependent, you also provide `∂t_uˢ` and `∂t_vˢ`.

### Example: Exponentially-decaying Stokes drift

The Stokes drift from a monochromatic deep-water surface wave decays exponentially with depth.
For a wave with wavelength ``\lambda`` propagating in the ``x``-direction, the Stokes drift is

```math
u^S(z) = U^S \exp(2 k z)
```

where ``k = 2\pi / \lambda`` is the wavenumber and ``U^S`` is the surface Stokes drift velocity.
The vertical derivative is ``\partial_z u^S = 2 k U^S \exp(2 k z)``.

```jldoctest stokes2D
using Oceananigans

wavelength = 60  # meters
wavenumber = 2π / wavelength
Uˢ = 0.01  # m/s, surface Stokes drift
∂z_uˢ(z, t) = 2 * wavenumber * Uˢ * exp(2 * wavenumber * z)
stokes_drift = UniformStokesDrift(∂z_uˢ = ∂z_uˢ)

# output

UniformStokesDrift{Nothing}:
├── ∂z_uˢ: ∂z_uˢ
├── ∂z_vˢ: zerofunction
├── ∂t_uˢ: zerofunction
└── ∂t_vˢ: zerofunction
```

### Example: Using parameters

Instead of defining global constants, you can pass a `parameters` NamedTuple. When `parameters` is
provided, the Stokes drift functions are called with the signature `(z, t, parameters)`:

```jldoctest stokes2D
∂z_uˢ(z, t, p) = 2 * p.k * p.Uˢ * exp(2 * p.k * z)
stokes_params = (k = 2π / 60, Uˢ = 0.01)
stokes_drift = UniformStokesDrift(∂z_uˢ = ∂z_uˢ, parameters = stokes_params)

# output

UniformStokesDrift with parameters (k=0.10472, Uˢ=0.01):
├── ∂z_uˢ: ∂z_uˢ
├── ∂z_vˢ: zerofunction
├── ∂t_uˢ: zerofunction
└── ∂t_vˢ: zerofunction
```

### Using `UniformStokesDrift` in a model

To include Stokes drift in a [`NonhydrostaticModel`](@ref), pass the `stokes_drift` object as a
keyword argument:

```jldoctest stokes2D
grid = RectilinearGrid(size = (32, 32, 32), extent = (128, 128, 64))
∂z_uˢ(z, t) = 0.01 * exp(z / 10)
model = NonhydrostaticModel(grid; stokes_drift = UniformStokesDrift(∂z_uˢ = ∂z_uˢ))
model.stokes_drift

# output

UniformStokesDrift{Nothing}:
├── ∂z_uˢ: ∂z_uˢ
├── ∂z_vˢ: zerofunction
├── ∂t_uˢ: zerofunction
└── ∂t_vˢ: zerofunction
```

## `StokesDrift`

[`StokesDrift`](@ref) is used for three-dimensional, spatially-varying Stokes drift fields such
as might arise from a wave packet with a horizontally-varying envelope:

```math
\boldsymbol{u}^S = u^S(x, y, z, t) \hat{\boldsymbol{x}} + v^S(x, y, z, t) \hat{\boldsymbol{y}} + w^S(x, y, z, t) \hat{\boldsymbol{z}}
```

!!! warning "Divergence-free requirement"
    For three-dimensional Stokes drift, the provided Stokes drift field must be divergence-free
    (solenoidal): ``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{u}^S = 0``. This is required for consistency with
    the Craik-Leibovich formulation as discussed by [VannesteYoung2022](@citet).
    In practice, this means that if the horizontal components of Stokes drift vary in space,
    there must be a corresponding vertical Stokes drift component ``w^S`` such that the
    total field is divergence-free.

To construct a [`StokesDrift`](@ref), you provide functions for the spatial and temporal derivatives
of the Stokes drift velocity components. The required derivatives are:

- `∂z_uˢ`, `∂y_uˢ`, `∂t_uˢ` for ``u^S``
- `∂z_vˢ`, `∂x_vˢ`, `∂t_vˢ` for ``v^S``
- `∂x_wˢ`, `∂y_wˢ`, `∂t_wˢ` for ``w^S``

### Example: A surface wave packet

Consider a wave packet propagating in the ``x``-direction with group velocity ``c^g``.
The Stokes drift can be written as

```math
u^S(x, y, z, t) = A(\xi, \eta) \, \hat{u}^S(z)
```

where ``\xi = x - c^g t``, ``\eta = y``, and ``A`` is a Gaussian envelope
``A(\xi, \eta) = \exp[-(\xi^2 + \eta^2) / 2\delta^2]``.

To satisfy the divergence-free condition ``\partial_x u^S + \partial_z w^S = 0``,
we must have

```math
w^S = -\frac{1}{2k} \partial_\xi A \, \hat{u}^S(z)
```

The following example sets up such a wave packet:

```jldoctest stokes3D
using Oceananigans
using Oceananigans.Units

g = 9.81
wavelength = 100meters
const k = 2π / wavelength
c = sqrt(g / k)
const δ = 400kilometers  # wave packet width
const cᵍ = c / 2         # group velocity
const Uˢ = 0.01          # m/s

# Envelope functions
@inline A(ξ, η) = exp(-(ξ^2 + η^2) / 2δ^2)
@inline ∂ξ_A(ξ, η) = -ξ / δ^2 * A(ξ, η)
@inline ∂η_A(ξ, η) = -η / δ^2 * A(ξ, η)
@inline ∂²ξ_A(ξ, η) = (ξ^2 / δ^2 - 1) * A(ξ, η) / δ^2
@inline ∂η_∂ξ_A(ξ, η) = ξ * η / δ^4 * A(ξ, η)

# Vertical structure
@inline ûˢ(z) = Uˢ * exp(2k * z)

# Stokes drift derivatives
@inline ∂z_uˢ(x, y, z, t) = 2k * A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂y_uˢ(x, y, z, t) = ∂η_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂t_uˢ(x, y, z, t) = -cᵍ * ∂ξ_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂x_wˢ(x, y, z, t) = -1 / 2k * ∂²ξ_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂y_wˢ(x, y, z, t) = -1 / 2k * ∂η_∂ξ_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂t_wˢ(x, y, z, t) = cᵍ / 2k * ∂²ξ_A(x - cᵍ * t, y) * ûˢ(z)

stokes_drift = StokesDrift(; ∂z_uˢ, ∂y_uˢ, ∂t_uˢ, ∂x_wˢ, ∂y_wˢ, ∂t_wˢ)

# output

StokesDrift{Nothing}:
├── ∂x_vˢ: zerofunction
├── ∂x_wˢ: ∂x_wˢ
├── ∂y_uˢ: ∂y_uˢ
├── ∂y_wˢ: ∂y_wˢ
├── ∂z_uˢ: ∂z_uˢ
├── ∂z_vˢ: zerofunction
├── ∂t_uˢ: ∂t_uˢ
├── ∂t_vˢ: zerofunction
└── ∂t_wˢ: ∂t_wˢ
```

### Function signatures

The function signature for [`StokesDrift`](@ref) depends on the grid topology:

- For a grid with `topology = (Periodic, Periodic, Bounded)` and `parameters = nothing`,
  functions are called as `f(x, y, z, t)`.
- For a grid with `topology = (Periodic, Flat, Bounded)` and `parameters = nothing`,
  functions are called as `f(x, z, t)` (the ``y`` coordinate is omitted).
- When `parameters` is provided, it is passed as an additional final argument:
  `f(x, y, z, t, parameters)` or `f(x, z, t, parameters)`.

