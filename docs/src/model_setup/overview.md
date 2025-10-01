# Models: discrete equations and state variables

Oceananigans serves two mature models: [NonhydrostaticModel](@ref), which solves the Navier-Stokes equations
under the Boussinesq approximation _without_ making the hydrostatic approximation, and [HydrostaticFreeSurfaceModel](@ref),
which solves the hydrostatic or "primitive" Boussinesq equations with a "free" surface on the top boundary.
A third, experimental [ShallowWaterModel](@ref) solves the shallow water equations.

The NonhydrostaticModel is primarily used for large eddy simulations on [RectilinearGrid](@ref) with grid spacings of O(1 m), but can also be used for idealized classroom problems (e.g. two-dimensional turbulence) and direct numerical simulation.
HydrostaticFreeSurfaceModel, on the other hand, derives its purpose at larger scales --- typically for regional to global simulations with grid spacings of O(30 m) and up, on [RectilinearGrid](@ref),
[LatitudeLongitudeGrid](@ref), [TripolarGrid](@ref), [ConformalCubedSphereGrid](@ref),
and other [OrthogonalSphericalShellGrid](@ref)s such as [RotatedLatitudeLongitudeGrid](@ref Oceananigans.OrthogonalSphericalShellGrids.RotatedLatitudeLongitudeGrid).

## Whence Models?

Oceananigans models may be distilled to two aspects: _(i)_ specification for a set of discrete equations, and
_(ii)_ a container for the prognostic and diagnostic state of those equations.

### Configuring models by changing keyword arguments

By specifying discrete equations, a model may be integrated or "stepped forward"
in time by calling `time_step!(model, Δt)`, where `Δt` is the time step
and thus advancing the `model.clock`.
The `time_step!` interface is used by [`Simulation`](@ref) to manage time-stepping
along with other activities, like monitoring progress, writing output to disk, and more.

To illustrate discrete equation specification, consider the docstring for `NonhydrostaticModel`:

```@docs
NonhydrostaticModel
```

```@example
using Oceananigans

arch = CPU()
grid = RectilinearGrid(arch,
                       size = (128, 128),
                       x = (0, 256),
                       z = (-128, 0),
                       topology = (Periodic, Flat, Bounded))

# Numerical method and physics choices
advection = WENO(order=9) # ninth‑order upwind for momentum and tracers
buoyancy = SeawaterBuoyancy()  # requires T, S tracers

τₓ = - 8e-5 # m² s⁻² (τₓ < 0 ⟹ eastward wind stress)
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τₓ))

# A small sinusoidal cooling tendency for T
c_source(x, y, z, t, p) = -p.μ * cos(2π * x) * exp(z / p.H)
c_forcing = Forcing(c_source; parameters=(μ=1e-3, H=0.2))

model = NonhydrostaticModel(; grid, advection, buoyancy,
                            tracers = (:T, :S, :c),
                            boundary_conditions = (; u=u_bcs),
                            forcing = (; c=c_forcing))
```

1. Specify the discrete equations to solve: physics options (buoyancy, Coriolis, free surface), numerical methods (advection schemes, closures), and configuration like forcing and boundary conditions.
   These are primarily set via keyword arguments when constructing a model, and many parameters can be adjusted later.
2. Hold the simulation state: prognostic state (velocities, tracers, pressure/free surface) and diagnostic/auxiliary fields.
   Every model pairs with `set!(model; kwargs...)` to update state any time.
   This is typically used for initial conditions, but can also be used to change state mid‑simulation.

We can advance a model in time with `time_step!(model, Δt)`.
However, we generally recommend using `Simulation` to manage time stepping (including adaptive time steps) and the output.
See the [Quick start](@ref quick_start) for a compact example.

## Two Model Flavors

Oceananigans provides multiple models. This tutorial focuses on two:

- `NonhydrostaticModel`: Solves Boussinesq, incompressible Navier–Stokes equations with nonhydrostatic pressure.
- `HydrostaticFreeSurfaceModel`: Solves Boussinesq equations under the hydrostatic approximation, with a prognostic free surface.

For the governing equations and details, see Physics pages for the [`NonhydrostaticModel`](@ref) and the [`HydrostaticFreeSurfaceModel`](@ref hydrostatic_free_surface_model).

### Constructor Reference

The docstrings below summarize the main constructor options. Later sections show compact examples.

```@docs
NonhydrostaticModel
```

```@docs
HydrostaticFreeSurfaceModel
```

```@docs
ShallowWaterModel
```

## Minimal Examples

Start with a simple box grid and build each model with sensible defaults.

```jldoctest
using Oceananigans

grid = RectilinearGrid(size=(8, 8, 8), extent=(1, 1, 1))

nh = NonhydrostaticModel(; grid)                 # no buoyancy or tracers by default
hy = HydrostaticFreeSurfaceModel(; grid)         # default free surface, no tracers

nh, hy

# output

(NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: Centered(order=2)
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing, HydrostaticFreeSurfaceModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
├── free surface: ImplicitFreeSurface with gravitational acceleration 9.80665 m s⁻²
│   └── solver: FFTImplicitFreeSurfaceSolver
├── advection scheme:
│   └── momentum: VectorInvariant
├── vertical_coordinate: ZCoordinate
└── coriolis: Nothing)
```

Both models create velocity fields and time‑steppers; the tracer sets start empty unless specified.

## Discrete Equations: Key Ingredients

This section illustrates how advection schemes, buoyancy, closures, forcing, and boundary conditions are configured at construction. The snippets are self‑contained and intended as patterns; see the Model setup pages for deeper options and examples.

### NonhydrostaticModel: advection, buoyancy, closure, forcing, boundary conditions

`NonhydrostaticModel` uses a single advection scheme for both momentum and tracers.

```jldoctest models_nh
using Oceananigans

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

# Numerical method and physics choices
advection = WENO()  # fifth‑order upwind for momentum and tracers
buoyancy = SeawaterBuoyancy()  # requires T, S tracers
closure = ScalarDiffusivity(ν=1e-6, κ=(T=1e-7, S=1e-7))

# Simple wind stress and surface cooling via boundary conditions + forcing
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions

ρ₀ = 1027.0   # kg m⁻³
τₓ = 0.08     # N m⁻²  (eastward wind stress)

u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(-τₓ / ρ₀))

# A small sinusoidal cooling tendency for T
T_cool(x, y, z, t, p) = -p.μ * cos(2π * x) * exp(z / p.H)
T_forcing = Forcing(T_cool; parameters=(μ=1e-3, H=0.2))

model = NonhydrostaticModel(; grid,
                            advection,
                            buoyancy,
                            tracers=(:T, :S),
                            closure,
                            boundary_conditions=(; u=u_bcs),
                            forcing=(; T=T_forcing))

model

# output

NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: WENO{3, Float64, Float32}(order=5)
├── tracers: (T, S)
├── closure: ScalarDiffusivity{ExplicitTimeDiscretization}(ν=1.0e-6, κ=(T=1.0e-7, S=1.0e-7))
├── buoyancy: SeawaterBuoyancy with g=9.80665 and LinearEquationOfState(thermal_expansion=0.000167, haline_contraction=0.00078) with ĝ = NegativeZDirection()
└── coriolis: Nothing
```

Notes
- Advection: [`WENO()`](@ref WENO) is a robust, high‑order upwind method for momentum and tracers.
- Buoyancy: `SeawaterBuoyancy()` activates Boussinesq buoyancy with a linear equation of state and gravity; it requires `:T` and `:S` tracers.
- Closure: `ScalarDiffusivity` sets molecular or eddy viscosities/diffusivities.
  See [Turbulence closures](@ref turbulence_closures) for alternatives like `SmagorinskyLilly()` or `AnisotropicMinimumDissipation()`.
- Forcing: `Forcing` functions can depend on `x, y, z, t` and parameters; see [Forcing functions](@ref forcing_functions) for field‑dependent and discrete forms.
- Boundary conditions: Here we add surface wind stress via a `FluxBoundaryCondition` on `u`.
  See [Boundary conditions](@ref model_step_bcs) for Value/Flux/Gradient forms and more patterns.

### HydrostaticFreeSurfaceModel: momentum vs tracer advection, buoyancy, closure, surface stress

`HydrostaticFreeSurfaceModel` separates momentum and tracer advection and evolves a prognostic free surface `η`.

```jldoctest model_hy
using Oceananigans
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

momentum_advection = VectorInvariant()   # recommended on curvilinear grids too
tracer_advection   = WENO()              # upwinded tracer advection

buoyancy = SeawaterBuoyancy()
closure  = ScalarDiffusivity(ν=1e-6, κ=(T=1e-7, S=1e-7))

ρ₀ = 1027.0
τₓ = 0.05
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(-τₓ / ρ₀))

model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection,
                                    tracer_advection,
                                    buoyancy,
                                    tracers=(:T, :S),
                                    closure,
                                    boundary_conditions=(; u=u_bcs))

model

# output
HydrostaticFreeSurfaceModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: (T, S)
├── closure: ScalarDiffusivity{ExplicitTimeDiscretization}(ν=1.0e-6, κ=(T=1.0e-7, S=1.0e-7))
├── buoyancy: SeawaterBuoyancy with g=9.80665 and LinearEquationOfState(thermal_expansion=0.000167, haline_contraction=0.00078) with ĝ = NegativeZDirection()
├── free surface: ImplicitFreeSurface with gravitational acceleration 9.80665 m s⁻²
│   └── solver: FFTImplicitFreeSurfaceSolver
├── advection scheme:
│   ├── momentum: VectorInvariant
│   ├── T: WENO{3, Float64, Float32}(order=5)
│   └── S: WENO{3, Float64, Float32}(order=5)
├── vertical_coordinate: ZCoordinate
└── coriolis: Nothing
```

**Notes**

- Momentum advection defaults to `VectorInvariant()`; tracer advection defaults to `Centered(order=2)`.
  Users may choose schemes independently.
- Hydrostatic models include a free surface; the default is an implicit free surface on regular rectilinear grids.
  See the [Hydrostatic physics page](@ref hydrostatic_free_surface_model) for details and generalized vertical coordinates.

## State: Initial conditions and updates with `set!`

All models support `set!(model; kwargs...)` to initialize or update fields.
`kwargs` can be:
- constant values,
- arrays, or
- functions of the grid's extrinsic coordinates, e.g., `(x, y, z)`.

### Nonhydrostatic initial condition (shear and stratification)

```jldoctest
using Oceananigans

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=(Periodic, Flat, Bounded))
model = NonhydrostaticModel(; grid, advection=WENO(), tracers=(:T, :S), buoyancy=SeawaterBuoyancy())

U₀ = 0.5
u₀(x, y, z) = U₀ * tanh(8z - 4)                # vertical shear
T₀(x, y, z) = 1 + 0.01 * z                     # stable stratification
S₀(x, y, z) = 35 + 0.0 * z

set!(model; u=u₀, T=T₀, S=S₀)

model.velocities.u, model.tracers.T

# output

(16×16×16 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 22×22×22 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, -2:19) with eltype Float64 with indices -2:19×-2:19×-2:19
    └── max=-0.499797, min=-0.5, mean=-0.49998, 16×16×16 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 22×22×22 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, -2:19) with eltype Float64 with indices -2:19×-2:19×-2:19
    └── max=0.999687, min=0.990313, mean=0.995)
```

!!! tip "Divergence-free velocity fields"
    For the NonhydrostaticModel, as part of the time-stepping algorithm, the velocity
    field is made divergence-free at every time step. So if a model is not initialized
    with a divergence-free velocity field, it may change on the first time step.
    As a result tracers may not be conserved up to machine precision at the first time step.

### Hydrostatic initial condition (surface displacement and currents)

`HydrostaticFreeSurfaceModel` also accepts `η` (free surface) in `set!`.

```jldoctest
using Oceananigans
using Oceananigans.Units

Lx = Ly = 10kilometers
Lz = 4000meters
grid = RectilinearGrid(size=(16, 16, 16), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))
model = HydrostaticFreeSurfaceModel(; grid, tracers=:b, buoyancy=BuoyancyTracer())

N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model; b=bᵢ)

model.tracers.b

# output

16×16×16 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 22×22×22 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, -2:19) with eltype Float64 with indices -2:19×-2:19×-2:19
    └── max=-0.00125, min=-0.03875, mean=-0.02
```

## Stepping and Simulations

You can advance a model with a single step:

```jldoctest
using Oceananigans

grid = RectilinearGrid(size=(8, 8, 8), extent=(1, 1, 1))
model = NonhydrostaticModel(; grid)

time_step!(model, 0.01)

model.clock.time > 0

# output

true
```

But for real simulations we recommend `Simulation` for running, output, and adaptive `Δt`. See [Quick start](@ref quick_start) and the Examples gallery for complete workflows, including Kelvin–Helmholtz instability and wind‑driven mixed layers.

## Where to go next

- Model setup (legacy): in‑depth pages on buoyancy, forcing, boundary conditions, closures, diagnostics, and output.
- Physics: governing equations and numerical forms for [`NonhydrostaticModel`](@ref) and [`HydrostaticFreeSurfaceModel`](@ref hydrostatic_free_surface_model).
- Examples: browse literated examples for richer end‑to‑end setups.
