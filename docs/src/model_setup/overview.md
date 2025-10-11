# Models

In general in Oceananigans, the "model" object serves two main purposes:
 _(i)_ models store the configuration of a set of discrete equations. The discrete equations imply rules for evolving prognostic variables, and computing diagnostic varaibles from the prognostic state.
 _(ii)_ models provide a container for the prognostic and diagnostic state of those discrete equations at a particular time.

## Two Oceananigans models for ocean simulations

In addition to defining the abstract concept of a "model" that can be used with [Simulation](@ref),
Oceananigans provides two mature model implementations for simulating ocean-flavored fluid dynamics.
Both of these integrate the Navier-Stokes equations within the Boussinesq approximation
(we call these the "Boussinesq equations" for short): the [NonhydrostaticModel](@ref) and the [HydrostaticFreeSurfaceModel](@ref).

The [NonhydrostaticModel](@ref) integrates the Boussinesq equations _without_ making the hydrostatic approximation,
and therefore possessing a prognostic vertical momentum equation. The NonhydrostaticModel is useful for simulations
that resolve three-dimensional turbulence, such as large eddy simulations on [RectilinearGrid](@ref) with grid spacings of O(1 m),
as well as direct numerical simulation. The NonhydrostaticModel may also be used for idealized classroom problems,
as in the [two-dimensional turbulence example](@ref "Two dimensional turbulence example").

The [HydrostaticFreeSurfaceModel](@ref) integrates the hydrostatic or "primitive" Boussinesq equations
with a free surface on its top boundary. The hydrostatic approximation allosw the HydrostaticFreeSurfaceModel
to achieve much higher efficiency in simulations on curvilinear grids used for large-scale regional or global simulations such as
[LatitudeLongitudeGrid](@ref), [TripolarGrid](@ref), [ConformalCubedSphereGrid](@ref),
and other [OrthogonalSphericalShellGrid](@ref)s such as [RotatedLatitudeLongitudeGrid](@ref Oceananigans.OrthogonalSphericalShellGrids.RotatedLatitudeLongitudeGrid).
Because they span larger domains, simulations with the HydrostaticFreeSurfaceModel also usually involve coarser grid spacings of O(30 m) up to O(100 km).
Such coarse-grained simulations are usually paired with more elaborate turublence closures or "parameterizations" than
small-scale simulations with NonhydrostaticModel, such as the vertical mixing schemes
[CATKEVerticalDiffusivity](@ref),
[RiBasedVerticalDiffusivity](@ref), and
[TKEDissipationVerticalDiffusivity](@ref), and the mesoscale turbulence closure
[IsopycnalSkewSymmetricDiffusivity](@ref) (a.k.a. "Gent-McWilliams plus Redi").

A third, experimental [ShallowWaterModel](@ref) solves the shallow water equations.

### Configuring NonhydrostaticModel with keyword arguments

To illustrate the specification of discrete equations, consider first the docstring for [NonhydrostaticModel](@ref),

```@docs
NonhydrostaticModel
```

The configuration operations for NonhydrostaticModel include "discretization choices", such as the `advection` scheme,
as well as aspects of the continuous underlying equations, such as the formulation of the buoyancy force.

For our first example, we build  the default `NonhydrostaticModel` (which is quite boring):

```@example
using Oceananigans
grid = RectilinearGrid(size=(8, 8, 8), extent=(8, 8, 8))
nh = NonhydrostaticModel(; grid)
```

The default `NonhydrostaticModel` has no tracers, no buoyancy force, no Coriolis force, and a second-order advection scheme.
We next consider a slightly more exciting NonhydrostaticModel configured with a WENO advection scheme,
the temperature/salinity-based [SeawaterBuoyancy](@ref), a boundary condition on the zonal momentum,
and a passive tracer forced by a cooked-up surface flux called "c":

```@example first_model
using Oceananigans

grid = RectilinearGrid(size=(128, 128), halo=(5, 5), x=(0, 256), z=(-128, 0),
                       topology = (Periodic, Flat, Bounded))

# Numerical method and physics choices
advection = WENO(order=9) # ninth‑order upwind for momentum and tracers
buoyancy = BuoyancyTracer()
coriolis = FPlane(f=1e-4)

τx = - 8e-5 # m² s⁻² (τₓ < 0 ⟹ eastward wind stress)
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τx))

@inline Jc(x, t, Lx) = cos(2π / Lx * x)
c_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Jc, parameters=grid.Lx))

model = NonhydrostaticModel(; grid, advection, buoyancy, coriolis,
                            tracers = (:b, :c),
                            boundary_conditions = (; u=u_bcs, c=c_bcs))
```

### Mutation of the model state

In addition to providing an interface for configuring equations, models also
store the prognostic and diagnostic state associated with the solution to those equations.
Models thus also provide an interface for "setting" or fixing the prognostic state, which is typically
invoked to determine the initial conditions of a simulation.
To illustrate this we consider setting the above model to a stably-stratified and noisy condition:

```@example first_model
N² = 1e-5
bᵢ(x, z) = N² * z + 1e-6 * randn()
uᵢ(x, z) = 1e-3 * randn()
set!(model, b=bᵢ, u=uᵢ, w=uᵢ)

model.tracers.b
```

Invoking `set!` above determine the model tracer `b` and the velocity components `u` and `w`.
`set!` also computes the diagnostic state of a model, which in the case of `NonhydrostaticModel` includes
the nonhydrostatic component of pressure,

```@example first_model
model.pressures.pNHS
```

### Evolving models in time

Model may be integrated or "stepped forward" in time by calling `time_step!(model, Δt)`, where `Δt` is the time step
and thus advancing the `model.clock`:

```@example first_model
time_step!(model, 1)
model.clock
```

However, users are strongly encouraged to use the [`Simulation`](@ref) interface to manage time-stepping
along with other activities, like monitoring progress, writing output to disk, and more.

```@example first_model
simulation = Simulation(model, Δt=1, stop_iteration=10)
run!(simulation)

simulation
```

## Using the HydrostaticFreeSurfaceModel

The HydrostaticFreeSurfaceModel has a similar interface as the NonhydrostaticModel,

```@example
using Oceananigans
grid = RectilinearGrid(size=(8, 8, 8), extent=(1, 1, 1))
model = HydrostaticFreeSurfaceModel(; grid) # default free surface, no tracers
```

```@example second_model
using Oceananigans
using SeawaterPolynomials: TEOS10EquationOfState

grid = LatitudeLongitudeGrid(size = (180, 80, 10),
                             longitude = (0, 360),
                             latitude = (-80, 80),
                             z = (-1000, 0),
                             halo = (6, 6, 3))

momentum_advection = WENOVectorInvariant()
coriolis = HydrostaticSphericalCoriolis()
equation_of_state = TEOS10EquationOfState()
buoyancy = SeawaterBuoyancy(; equation_of_state)
closure = CATKEVerticalDiffusivity()

# Generate a zonal wind stress that mimics Earth's mean winds
# with westerlies in mid-latitudes and easterlies near equator and poles
function zonal_wind_stress(λ, φ, t)
    # Parameters
    τ₀ = 1e-4  # Maximum wind stress magnitude (N/m²)
    φ₀ = 30   # Latitude of maximum westerlies (degrees)
    dφ = 10

    # Approximate wind stress pattern
    return - τ₀ * (+ exp(-(φ - φ₀)^2 / 2dφ^2)
                   - exp(-(φ + φ₀)^2 / 2dφ^2)
                   - 0.3 * exp(-φ^2 / dφ^2))
end

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(zonal_wind_stress))

model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, coriolis, closure, buoyancy,
                                    boundary_conditions = (; u=u_bcs), tracers=(:T, :S, :e))
```

Mutating the state of the HydrostaticFreeSurfaceModel works similarly as for the NonhydrostaticModel ---
except that the vertical velocity cannot be `set!`, because vertical velocity is not
prognostic in the hydrostatic equations.

```@example second_model
using SeawaterPolynomials

N² = 1e-5
T₀ = 20
S₀ = 35
eos = model.buoyancy.formulation.equation_of_state
α = SeawaterPolynomials.thermal_expansion(T₀, S₀, 0, eos)
g = model.buoyancy.formulation.gravitational_acceleration
dTdz = N² / (α * g)
Tᵢ(λ, φ, z) = T₀ + dTdz * z + 1e-3 * T₀ * randn()
uᵢ(λ, φ, z) = 1e-3 * randn()
set!(model, T=Tᵢ, S=S₀, u=uᵢ, v=uᵢ)

model.tracers.T
```

## Where to go next

- See [Quick start](@ref quick_start) for a compact, end-to-end workflow
- See the Examples gallery for longer tutorials covering specific cases, including large eddy simulation, Kelvin–Helmholtz instability and baroclinic instability.
- Other pages in the Models section: in‑depth pages on buoyancy, forcing, boundary conditions, closures, diagnostics, and output.
- Physics: governing equations and numerical forms for [`NonhydrostaticModel`](@ref) and [`HydrostaticFreeSurfaceModel`](@ref hydrostatic_free_surface_model).
