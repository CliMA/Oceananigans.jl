# Forcing functions

"Forcings" are user-defined terms appended to right-hand side of
the momentum or tracer evolution equations. In `Oceananigans`, momentum
and tracer forcings are defined via julia functions. `Oceananigans` includes
an interface for implementing forcing functions that depend on spatial coordinates,
time, model velocity and tracer fields, and external parameters.

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

Forcings are added to `Oceananigans` models by passing a `NamedTuple` of functions
or forcing objects to the `forcing` keyword argument in `NonhydrostaticModel`'s constructor.
By default, momentum and tracer forcing functions are assumed to be functions of
`x, y, z, t`. A basic example is

```jldoctest
u_forcing(x, y, z, t) = exp(z) * cos(x) * sin(t)

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, forcing=(u=u_forcing,))

model.forcing.u

# output
ContinuousForcing{Nothing} at (Face, Center, Center)
├── func: u_forcing (generic function with 1 method)
├── parameters: nothing
└── field dependencies: ()
```

More general forcing functions are built via the `Forcing` constructor
described below. `Oceananigans` also provides two convenience types:
    * `Relaxation` for damping terms that restore a field to a
      target distribution outside of a masked region of space. `Relaxation` can be
      used to implement sponge layers near the boundaries of a domain.
    * `AdvectiveForcing` for advecting individual quantities by a separate or
      "slip" velocity relative to both the prognostic model velocity field and any
      `BackgroundField` velocity field.

## The `Forcing` constructor

The `Forcing` constructor provides an interface for specifying forcing functions that

1. Depend on external parameters; and
2. Depend on model fields at the `x, y, z` location that forcing is applied; and/or
3. Require access to discrete model data.

### Forcing functions with external parameters

Most forcings involve external, changeable parameters.
Here are two examples of `forcing_func`tions that depend on 
_(i)_ a single scalar parameter `s`, and _(ii)_ a `NamedTuple` of parameters, `p`:

```jldoctest parameterized_forcing
# Forcing that depends on a scalar parameter `s`
u_forcing_func(x, y, z, t, s) = s * z

u_forcing = Forcing(u_forcing_func, parameters=0.1)

# Forcing that depends on a `NamedTuple` of parameters `p`
T_forcing_func(x, y, z, t, p) = - p.μ * exp(z / p.λ) * cos(p.k * x) * sin(p.ω * t)

T_forcing = Forcing(T_forcing_func, parameters=(μ=1, λ=0.5, k=2π, ω=4π))

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, forcing=(u=u_forcing, T=T_forcing), buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))

model.forcing.T

# output
ContinuousForcing{NamedTuple{(:μ, :λ, :k, :ω), Tuple{Int64, Float64, Float64, Float64}}} at (Center, Center, Center)
├── func: T_forcing_func (generic function with 1 method)
├── parameters: (μ = 1, λ = 0.5, k = 6.283185307179586, ω = 12.566370614359172)
└── field dependencies: ()
```

```jldoctest parameterized_forcing
model.forcing.u

# output
ContinuousForcing{Float64} at (Face, Center, Center)
├── func: u_forcing_func (generic function with 1 method)
├── parameters: 0.1
└── field dependencies: ()
```

In this example, the objects passed to the `parameters` keyword in the construction of
`u_forcing` and `T_forcing` --- a floating point number for `u_forcing`, and a `NamedTuple`
of parameters for `T_forcing` --- are passed on to `u_forcing_func` and `T_forcing_func` when
they are called during time-stepping. The object passed to `parameters` is in principle arbitrary.
However, if using the GPU, then `typeof(parameters)` may be restricted by the requirements
of GPU-compiliability.

### Forcing functions that depend on model fields

Forcing functions may depend on model fields (velocity, tracers or auxiliary fields) evaluated at the `x, y, z` where forcing is applied.
Here's a somewhat non-sensical example:

```jldoctest field_dependent_forcing
# Forcing that depends on the velocity fields `u`, `v`, and `w`
w_forcing_func(x, y, z, t, u, v, w) = - (u^2 + v^2 + w^2) / 2

w_forcing = Forcing(w_forcing_func, field_dependencies=(:u, :v, :w))

# Forcing that depends on salinity `S` and a scalar parameter
S_forcing_func(x, y, z, t, S, μ) = - μ * S

S_forcing = Forcing(S_forcing_func, parameters=0.01, field_dependencies=:S)

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, forcing=(w=w_forcing, S=S_forcing), buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))

model.forcing.w

# output
ContinuousForcing{Nothing} at (Center, Center, Face)
├── func: w_forcing_func (generic function with 1 method)
├── parameters: nothing
└── field dependencies: (:u, :v, :w)
```

```jldoctest field_dependent_forcing
model.forcing.S

# output
ContinuousForcing{Float64} at (Center, Center, Center)
├── func: S_forcing_func (generic function with 1 method)
├── parameters: 0.01
└── field dependencies: (:S,)
```

The `field_dependencies` arguments follow `x, y, z, t` in the forcing `func`tion in
the order they are specified in `Forcing`.
If both `field_dependencies` and `parameters` are specified, then the `field_dependencies`
arguments follow `x, y, z, t`, and `parameters` follow `field_dependencies`.

Model fields that arise in the arguments of continuous `Forcing` `func`tions are
automatically interpolated to the staggered grid location at which the forcing is applied.

### "Discrete form" forcing functions

"Discrete form" forcing functions are either called with the signature

```julia
func(i, j, k, grid, clock, model_fields)
```

or the parameterized form

```julia
func(i, j, k, grid, clock, model_fields, parameters)
```

Discrete form forcing functions can access the entirety of model field
data through the argument `model_fields`. The object `model_fields` is a `NamedTuple`
whose properties include the velocity fields `model_fields.u`, `model_fields.v`,
`model_fields.w` and all fields in `model.tracers`.

Using discrete forcing functions may require understanding the
staggered arrangement of velocity fields and tracers in `Oceananigans`.
Here's a slightly non-sensical example in which the vertical derivative of a buoyancy
tracer is used as a time-scale for damping the u-velocity field:

```jldoctest discrete_forcing
# A damping term that depends on a "local average":
local_average(i, j, k, grid, c) = @inbounds (c[i, j, k] + c[i-1, j, k] + c[i+1, j, k] +
                                                          c[i, j-1, k] + c[i, j+1, k] +
                                                          c[i, j, k-1] + c[i, j, k+1]) / 7

b_forcing_func(i, j, k, grid, clock, model_fields) = - local_average(i, j, k, grid, model_fields.b)

b_forcing = Forcing(b_forcing_func, discrete_form=true)

# A term that damps the local velocity field in the presence of stratification
using Oceananigans.Operators: ∂zᶠᶜᶠ, ℑxzᶠᵃᶜ

function u_forcing_func(i, j, k, grid, clock, model_fields, ε)
    # The vertical derivative of buoyancy, interpolated to the u-velocity location:
    N² = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, model_fields.b)

    # Set to zero in unstable stratification where N² < 0:
    N² = max(N², zero(typeof(N²)))

    return @inbounds - ε * sqrt(N²) * model_fields.u[i, j, k]
end

u_forcing = Forcing(u_forcing_func, discrete_form=true, parameters=1e-3)

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, tracers=:b, buoyancy=BuoyancyTracer(), forcing=(u=u_forcing, b=b_forcing))

model.forcing.b

# output
DiscreteForcing{Nothing}
├── func: b_forcing_func (generic function with 1 method)
└── parameters: nothing
```

```jldoctest discrete_forcing
model.forcing.u

# output
DiscreteForcing{Float64}
├── func: u_forcing_func (generic function with 1 method)
└── parameters: 0.001
```

The annotation `@inbounds` is crucial for performance when accessing array indices
of the fields in `model_fields`.

## `Relaxation`

`Relaxation` defines a special forcing function that restores a field at a specified `rate` to
a `target` distribution, within a region uncovered by a `mask`ing function.
`Relaxation` is useful for implementing sponge layers, as shown in the second example.

The following code constructs a model in which all components
of the velocity field are damped to zero everywhere on a time-scale of 1000 seconds, or ~17 minutes:

```jldoctest
damping = Relaxation(rate = 1/1000)

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)) 
model = NonhydrostaticModel(grid=grid, forcing=(u=damping, v=damping, w=damping))

model.forcing.w

# output
ContinuousForcing{Nothing} at (Center, Center, Face)
├── func: Relaxation(rate=0.001, mask=1, target=0)
├── parameters: nothing
└── field dependencies: (:w,)
```

The constructor for `Relaxation` accepts the keyword arguments `mask`, and `target`,
which specify a `mask(x, y, z)` function that multiplies the forcing, and a `target(x, y, z)`
distribution for the quantity in question. By default, `mask` uncovered the whole domain
and `target` restores the field in question to 0

We illustrate usage of `mask` and `target` by implementing a sponge layer that relaxes
velocity fields to zero and restores temperature to a linear gradient in the bottom
1/10th of the domain:

```jldoctest sponge_layer
grid = RectilinearGrid(size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(-1, 0))

        damping_rate = 1/100 # relax fields on a 100 second time-scale
temperature_gradient = 0.001 # ⁰C m⁻¹
 surface_temperature = 20    # ⁰C (at z=0)

target_temperature = LinearTarget{:z}(intercept=surface_temperature, gradient=temperature_gradient)
       bottom_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

uvw_sponge = Relaxation(rate=damping_rate, mask=bottom_mask)
  T_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=target_temperature)

model = NonhydrostaticModel(grid=grid, forcing=(u=uvw_sponge, v=uvw_sponge, w=uvw_sponge, T=T_sponge), buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))

model.forcing.u

# output
ContinuousForcing{Nothing} at (Face, Center, Center)
├── func: Relaxation(rate=0.01, mask=exp(-(z + 1.0)^2 / (2 * 0.1^2)), target=0)
├── parameters: nothing
└── field dependencies: (:u,)
```

```jldoctest sponge_layer
model.forcing.T

# output
ContinuousForcing{Nothing} at (Center, Center, Center)
├── func: Relaxation(rate=0.01, mask=exp(-(z + 1.0)^2 / (2 * 0.1^2)), target=20.0 + 0.001 * z)
├── parameters: nothing
└── field dependencies: (:T,)
```

## `AdvectiveForcing`

`AdvectiveForcing` defines a forcing function that represents advection by
a separate or "slip" velocity relative to the prognostic model velocity field.
`AdvectiveForcing` is implemented with native Oceananigans advection operators,
which means that tracers advected by the "flux form" advection term
``𝛁⋅𝐮_{\rm slip} c``. Caution is advised when ``𝐮_{\rm slip}`` is not divergence free.

As an example, consider a model for sediment settling at a constant rate:

```jldoctest
using Oceananigans

r_sediment = 1e-4 # [m] "Fine sand"
ρ_sediment = 1200 # kg m⁻³
ρ_ocean = 1026 # kg m⁻³
Δb = 9.81 * (ρ_ocean - ρ_sediment) / ρ_ocean # m s⁻²
ν_molecular = 1.05e-6 # m² s⁻¹
w_sediment = 2/9 * Δb / ν_molecular * r_sediment^2 # m s⁻¹

sinking = AdvectiveForcing(UpwindBiasedFifthOrder(), w=w_sediment)

# output
AdvectiveForcing with the UpwindBiased scheme:
├── u: ZeroField{Int64}
├── v: ZeroField{Int64}
└── w: ConstantField(-0.00352102)
```

The first argument to `AdvectiveForcing` is the advection scheme (here `UpwindBiasedFifthOrder()`).
The three keyword arguments specify the `u`, `v`, and `w` components of the separate
slip velocity field. The default for each `u, v, w` is `ZeroField`.

Next we consider a dynamically-evolving slip velocity. For this we use `ZFaceField`
with appropriate boundary conditions as our slip velocity:

```jldoctest sinking
using Oceananigans
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition

grid = RectilinearGrid(size=(32, 32, 32), x=(-10, 10), y=(-10, 10), z=(-4, 4),
                       topology=(Periodic, Periodic, Bounded))

no_penetration = ImpenetrableBoundaryCondition()
slip_bcs = FieldBoundaryConditions(grid, (Center, Center, Face),
                                   top=no_penetration, bottom=no_penetration)

w_slip = ZFaceField(grid, boundary_conditions=slip_bcs)
sinking = AdvectiveForcing(WENO(; grid), w=w_slip)

# output
AdvectiveForcing with the WENO scheme:
├── u: ZeroField{Int64}
├── v: ZeroField{Int64}
└── w: 32×32×33 Field{Center, Center, Face} on RectilinearGrid on CPU
```

To compute the slip velocity, we must add a `Callback`to `simulations.callback` that
computes `w_slip` ever iteration:

```jldoctest sinking
using Oceananigans.BoundaryConditions: fill_halo_regions!

model = NonhydrostaticModel(; grid, tracers=(:b, :P), forcing=(; P=sinking))
simulation = Simulation(model; Δt=1, stop_iteration=100)

# Build abstract operation for slip velocity
b_particle = - 1e-4 # relative buoyancy depends on reference density and initial buoyancy condition
b = model.tracers.b
R = 1e-3 # [m] mean particle radius
ν = 1.05e-6 # [m² s⁻¹] molecular kinematic viscosity of water
w_slip_op = 2/9 * (b - b_particle) / ν * R^2 # Stokes terminal velocity

function compute_slip_velocity!(sim)
    w_slip .= w_slip_op
    fill_halo_regions!(w_slip)
    return nothing
end

simulation.callbacks[:slip] = Callback(compute_slip_velocity!)

# output
Callback of compute_slip_velocity! on IterationInterval(1)
```
