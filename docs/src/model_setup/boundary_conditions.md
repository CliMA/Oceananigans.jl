# [Boundary conditions](@id model_step_bcs)

Boundary conditions are intimately related to the grid topology, and only
need to be considered in directions with `Bounded` topology.
In `Bounded` directions, tracer and momentum fluxes are conservative or "zero flux"
by default. Non-default boundary conditions are therefore required to specify non-zero fluxes
of tracers and momentum across `Bounded` directions, and across immersed boundaries
when using `ImmersedBoundaryGrid`.

See [Numerical implementation of boundary conditions](@ref numerical_bcs) for more details.

### Example: no-slip conditions on every boundary

```@meta
DocTestSetup = quote
   using Oceananigans

   using Random
   Random.seed!(1234)
end
```

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(16, 16, 16), x=(0, 2π), y=(0, 1), z=(0, 1), topology=(Periodic, Bounded, Bounded))
16×16×16 RectilinearGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [1.26883e-16, 6.28319) regularly spaced with Δx=0.392699
├── Bounded  y ∈ [0.0, 1.0]             regularly spaced with Δy=0.0625
└── Bounded  z ∈ [0.0, 1.0]             regularly spaced with Δz=0.0625

julia> no_slip_bc = ValueBoundaryCondition(0)
ValueBoundaryCondition: 0
```

A "no-slip" boundary condition that velocity components tangential to `Bounded`
directions decay to `0` at the boundary, leading to a viscous loss of momentum.

```jldoctest
julia> no_slip_field_bcs = FieldBoundaryConditions(no_slip_bc);

julia> model = NonhydrostaticModel(; grid, boundary_conditions=(u=no_slip_field_bcs, v=no_slip_field_bcs, w=no_slip_field_bcs))
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing

julia> model.velocities.u.boundary_conditions
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: PeriodicBoundaryCondition
├── east: PeriodicBoundaryCondition
├── south: ValueBoundaryCondition: 0
├── north: ValueBoundaryCondition: 0
├── bottom: ValueBoundaryCondition: 0
├── top: ValueBoundaryCondition: 0
└── immersed: FluxBoundaryCondition: Nothing
```

Boundary conditions are passed to `FieldBoundaryCondition` to build boundary conditions for each
field individually, and then onto the model constructor (here `NonhydrotaticModel`) via the 
keyword argument `boundary_conditions`.
The model constructor then "interprets" the input and builds appropriate boundary conditions
for the grid `topology`, given the user-specified `no_slip` default boundary condition for `Bounded`
directions. In the above example, note that the `west` and `east` boundary conditions are `PeriodicBoundaryCondition`
because the `x`-topology of the grid is `Periodic`.

### Example: specifying boundary conditions on individual boundaries

To specify no-slip boundary conditions on every `Bounded` direction _except_
the surface, we write

```jldoctest
julia> free_slip_surface_bcs = FieldBoundaryConditions(no_slip_bc, top=FluxBoundaryCondition(nothing));

julia> model = NonhydrostaticModel(; grid, boundary_conditions=(u=free_slip_surface_bcs, v=free_slip_surface_bcs, w=no_slip_field_bcs));

julia> model.velocities.u.boundary_conditions
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: PeriodicBoundaryCondition
├── east: PeriodicBoundaryCondition
├── south: ValueBoundaryCondition: 0
├── north: ValueBoundaryCondition: 0
├── bottom: ValueBoundaryCondition: 0
├── top: FluxBoundaryCondition: Nothing
└── immersed: FluxBoundaryCondition: Nothing

julia> model.velocities.v.boundary_conditions
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: PeriodicBoundaryCondition
├── east: PeriodicBoundaryCondition
├── south: OpenBoundaryCondition: Nothing
├── north: OpenBoundaryCondition: Nothing
├── bottom: ValueBoundaryCondition: 0
├── top: FluxBoundaryCondition: Nothing
└── immersed: FluxBoundaryCondition: Nothing
```

Now both `u` and `v` have `FluxBoundaryCondition(nothing)` at the `top` boundary, which is `Oceananigans` lingo
for "no-flux boundary condition".

## Boundary condition classifications

There are three primary boundary condition classifications:

1. [`FluxBoundaryCondition`](@ref) specifies fluxes directly.

   For example, sunlight absorbed at the ocean surface imparts a temperature flux that heats near-surface fluid.
   If there is a known `diffusivity`, you can express `FluxBoundaryCondition(flux)`
   using `GradientBoundaryCondition(-flux / diffusivity)` (aka "Neumann" boundary condition).
   But when `diffusivity` is not known or is variable (as for large eddy simulation, for example),
   it's convenient and more straightforward to apply `FluxBoundaryCondition`.

2. [`ValueBoundaryCondition`](@ref) (Dirchlet) specifies the value of a field on
   the given boundary, which when used in combination with a turbulence closure
   results in a flux across the boundary. For example, `ValueBoundaryCondition(0)` is used
   to specify no-slip boundary conditions on velocity components tangential to the given boundary.

   Examples with `ValueBoundaryCondition`:

   * Prescribe a surface to have a constant temperature, like 20 degrees.
     Heat will then flux in and out of the domain depending on the temperature difference between the surface and the interior, and the temperature diffusivity.
   * Prescribe a velocity tangent to a boundary as in a driven-cavity flow (for example), where the top boundary is moving.
     Momentum will flux into the domain do the difference between the top boundary velocity and the interior velocity, and the prescribed viscosity.

   _Note_: Do not use `ValueBoundaryCondition` on a wall-normal velocity component.
   `ImpenetrableBoundaryCondition` is internally enforced and thus only needs to be specified
   for "additional" fields created outside model constructors.

3. [`GradientBoundaryCondition`](@ref) (Neumann) specifies the gradient of a field on a boundary.

In addition to these primary boundary conditions, `ImpenetrableBoundaryCondition` applies to velocity
components in wall-normal directions.
Note however that impenetrability is internally enforced for model velocity components,
so that `ImpenetrableBoundaryCondition` is only used for _additional_ velocity components
that are not evolved by a model, such as a velocity component used for (`AdvectiveForcing`)[@ref].

Finally, note that `Periodic` boundary conditions are internally enforced for `Periodic` directions,
and `DefaultBoundaryConditions` may exist before boundary conditions are "materialized" by a model.

## Default boundary conditions

The default boundary condition in `Bounded` directions is no-flux, or `FluxBoundaryCondition(nothing)`.
The default boundary condition can be changed by passing a positional argument to `FieldBoundaryConditions`,
as in

```jldoctest
julia> no_slip_bc = ValueBoundaryCondition(0)
ValueBoundaryCondition: 0

julia> free_slip_surface_bcs = FieldBoundaryConditions(no_slip_bc, top=FluxBoundaryCondition(nothing))
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: DefaultBoundaryCondition (ValueBoundaryCondition: 0)
├── east: DefaultBoundaryCondition (ValueBoundaryCondition: 0)
├── south: DefaultBoundaryCondition (ValueBoundaryCondition: 0)
├── north: DefaultBoundaryCondition (ValueBoundaryCondition: 0)
├── bottom: DefaultBoundaryCondition (ValueBoundaryCondition: 0)
├── top: FluxBoundaryCondition: Nothing
└── immersed: DefaultBoundaryCondition (ValueBoundaryCondition: 0)
```

## Boundary condition structures

Oceananigans uses a hierarchical structure to express boundary conditions:

1. Each boundary has one [`BoundaryCondition`](@ref)
2. Each field has seven [`BoundaryCondition`](@ref) (`west`, `east`, `south`, `north`, `bottom`, `top` and
   `immersed`)
3. A set of `FieldBoundaryConditions`, up to one for each field, are grouped into a `NamedTuple` and passed
   to the model constructor.

## Specifying boundary conditions for a model

Boundary conditions are defined at model construction time by passing a `NamedTuple` of `FieldBoundaryConditions`
specifying non-default boundary conditions for fields such as velocities and tracers.

Fields for which boundary conditions are not specified are assigned a default boundary conditions.

A few illustrations are provided below. See the examples for
further illustrations of boundary condition specification.

## Creating individual boundary conditions with `BoundaryCondition`

Boundary conditions may be specified with constants, functions, or arrays.
In this section we illustrate usage of the different [`BoundaryCondition`](@ref) constructors.

### 1. Constant `Value` (Dirchlet) boundary condition

```jldoctest bcs
julia> constant_T_bc = ValueBoundaryCondition(20.0)
ValueBoundaryCondition: 20.0
```

A constant [`Value`](@ref) boundary condition can be used to specify constant tracer (such as temperature),
or a constant _tangential_ velocity component at a boundary. Note that boundary conditions on the
_normal_ velocity component must use the [`Open`](@ref) boundary condition type.

Finally, note that `ValueBoundaryCondition(condition)` is an alias for `BoundaryCondition(Value, condition)`.

### 2. Constant `Flux` boundary condition

```jldoctest
julia> ρ₀ = 1027;  # Reference density [kg/m³]

julia> τₓ = 0.08;  # Wind stress [N/m²]

julia> wind_stress_bc = FluxBoundaryCondition(-τₓ/ρ₀)
FluxBoundaryCondition: -7.78968e-5
```

A constant [`Flux`](@ref) boundary condition can be imposed on tracers and tangential velocity components
that can be used, for example, to specify cooling, heating, evaporation, or wind stress at the ocean surface.

!!! info "The flux convention in Oceananigans"
    `Oceananigans` uses the convention that positive fluxes produce transport in the
    _positive_ direction (east, north, and up for ``x``, ``y``, ``z``).
    This means, for example, that a _negative_ flux of momentum or velocity at a _top_
    boundary, such as in the above example, produces currents in the _positive_ direction,
    because it prescribes a downwards flux of momentum into the domain from the top.
    Likewise, a _positive_ temperature flux at the top boundary
    causes _cooling_, because it transports heat _upwards_, out of the domain.
    Conversely, a positive flux at a _bottom_ boundary acts to increase the interior
    values of a quantity.

### 3. Spatially- and temporally-varying flux

Boundary conditions may be specified by functions,

```jldoctest
julia> @inline surface_flux(x, y, t) = cos(2π * x) * cos(t);

julia> top_tracer_bc = FluxBoundaryCondition(surface_flux)
FluxBoundaryCondition: ContinuousBoundaryFunction surface_flux at (Nothing, Nothing, Nothing)
```

!!! info "Boundary condition functions"
    By default, a function boundary condition is called with the signature
    ```julia
    f(ξ, η, t)
    ```
    where `t` is time and `ξ, η` are spatial coordinates that vary along the boundary:
    * `f(y, z, t)` on `x`-boundaries;
    * `f(x, z, t)` on `y`-boundaries;
    * `f(x, y, t)` on `z`-boundaries.
    Alternative function signatures are specified by keyword arguments to
    `BoundaryCondition`, as illustrated in subsequent examples.

### 4. Spatially- and temporally-varying flux with parameters

Boundary condition functions may be 'parameterized',

```jldoctest
julia> @inline wind_stress(x, y, t, p) = - p.τ * cos(p.k * x) * cos(p.ω * t); # function with parameters

julia> top_u_bc = FluxBoundaryCondition(wind_stress, parameters=(k=4π, ω=3.0, τ=1e-4))
FluxBoundaryCondition: ContinuousBoundaryFunction wind_stress at (Nothing, Nothing, Nothing)
```

!!! info "Boundary condition functions with parameters"
    The keyword argument `parameters` above specifies that `wind_stress` is called
    with the signature `wind_stress(x, y, t, parameters)`. In principle, `parameters` is arbitrary.
    However, relatively simple objects such as floating point numbers or `NamedTuple`s must be used
    when running on the GPU.

### 5. 'Field-dependent' boundary conditions

Boundary conditions may also depend on model fields. For example, a linear drag boundary condition
is implemented with

```jldoctest
julia> @inline linear_drag(x, y, t, u) = - 0.2 * u
linear_drag (generic function with 1 method)

julia> u_bottom_bc = FluxBoundaryCondition(linear_drag, field_dependencies=:u)
FluxBoundaryCondition: ContinuousBoundaryFunction linear_drag at (Nothing, Nothing, Nothing)
```

`field_dependencies` specifies the name of the dependent fields either with a `Symbol` or `Tuple` of `Symbol`s.

### 6. 'Field-dependent' boundary conditions with parameters

When boundary conditions depends on fields _and_ parameters, their functions take the form

```jldoctest
julia> @inline quadratic_drag(x, y, t, u, v, drag_coeff) = - drag_coeff * u * sqrt(u^2 + v^2)
quadratic_drag (generic function with 1 method)

julia> u_bottom_bc = FluxBoundaryCondition(quadratic_drag, field_dependencies=(:u, :v), parameters=1e-3)
FluxBoundaryCondition: ContinuousBoundaryFunction quadratic_drag at (Nothing, Nothing, Nothing)
```

Put differently, `ξ, η, t` come first in the function signature, followed by field dependencies,
followed by `parameters` is `!isnothing(parameters)`.

### 7. Discrete-form boundary condition with parameters

Discrete field data may also be accessed directly from boundary condition functions
using the `discrete_form`. For example:

```jldoctest
@inline filtered_drag(i, j, grid, clock, model_fields) =
   @inbounds - 0.05 * (model_fields.u[i-1, j, 1] + 2 * model_fields.u[i, j, 1] + model_fields.u[i-1, j, 1])

u_bottom_bc = FluxBoundaryCondition(filtered_drag, discrete_form=true)

# output
FluxBoundaryCondition: DiscreteBoundaryFunction with filtered_drag
```

!!! info "The 'discrete form' for boundary condition functions"
    The argument `discrete_form=true` indicates to [`BoundaryCondition`](@ref) that `filtered_drag`
    uses the 'discrete form'. Boundary condition functions that use the 'discrete form'
    are called with the signature
    ```julia
    f(i, j, grid, clock, model_fields)
    ```
    where `i, j` are grid indices that vary along the boundary, `grid` is `model.grid`,
    `clock` is the `model.clock`, and `model_fields` is a `NamedTuple`
    containing `u, v, w` and the fields in `model.tracers`.
    The signature is similar for ``x`` and ``y`` boundary conditions expect that `i, j` is replaced
    with `j, k` and `i, k` respectively.

### 8. Discrete-form boundary condition with parameters

```jldoctest
julia> Cd = 0.2;  # drag coefficient

julia> @inline linear_drag(i, j, grid, clock, model_fields, Cd) = @inbounds - Cd * model_fields.u[i, j, 1];

julia> u_bottom_bc = FluxBoundaryCondition(linear_drag, discrete_form=true, parameters=Cd)
FluxBoundaryCondition: DiscreteBoundaryFunction linear_drag with parameters 0.2
```

!!! info "Inlining and avoiding bounds-checking in boundary condition functions"
    Boundary condition functions should be decorated with `@inline` when running on CPUs for performance reasons.
    On the GPU, all functions are force-inlined by default.
    In addition, the annotation `@inbounds` should be used when accessing the elements of an array
    in a boundary condition function (such as `model_fields.u[i, j, 1]` in the above example).
    Using `@inbounds` will avoid a relatively expensive check that the index `i, j, 1` is 'in bounds'.

### 9. A random, spatially-varying, constant-in-time temperature flux specified by an array

```jldoctest
julia> Nx = Ny = 16;  # Number of grid points.

julia> Q = randn(Nx, Ny); # temperature flux

julia> white_noise_T_bc = FluxBoundaryCondition(Q)
FluxBoundaryCondition: 16×16 Matrix{Float64}
```

When running on the GPU, `Q` must be converted to a `CuArray`.

## Building boundary conditions on a field

To create a set of [`FieldBoundaryConditions`](@ref) for a temperature field,
we write

```jldoctest
julia> T_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(20),
                                       bottom = GradientBoundaryCondition(0.01))
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: DefaultBoundaryCondition
├── east: DefaultBoundaryCondition
├── south: DefaultBoundaryCondition
├── north: DefaultBoundaryCondition
├── bottom: GradientBoundaryCondition: 0.01
├── top: ValueBoundaryCondition: 20
└── immersed: DefaultBoundaryCondition
```

If the grid is, e.g., horizontally-periodic, then each horizontal `DefaultPrognosticFieldBoundaryCondition`
is converted to `PeriodicBoundaryCondition` inside the model's constructor, before assigning the
boundary conditions to temperature `T`.

In general, boundary condition defaults are inferred from the field location and `topology(grid)`.

## Specifying model boundary conditions

To specify non-default boundary conditions, a named tuple of [`FieldBoundaryConditions`](@ref) objects is
passed to the keyword argument `boundary_conditions` in the [`NonhydrostaticModel`](@ref) constructor.
The keys of `boundary_conditions` indicate the field to which the boundary condition is applied.
Below, non-default boundary conditions are imposed on the ``u``-velocity and temperature.

```jldoctest
julia> topology = (Periodic, Periodic, Bounded);

julia> grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=topology);

julia> u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(+0.1),
                                       bottom = ValueBoundaryCondition(-0.1));

julia> c_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(20),
                                       bottom = GradientBoundaryCondition(0.01));

julia> model = NonhydrostaticModel(grid=grid, boundary_conditions=(u=u_bcs, c=c_bcs), tracers=:c)
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: c
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing

julia> model.velocities.u
16×16×16 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: Value, top: Value, immersed: ZeroFlux
└── data: 22×22×22 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, -2:19) with eltype Float64 with indices -2:19×-2:19×-2:19
    └── max=0.0, min=0.0, mean=0.0

julia> model.tracers.c
16×16×16 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: Gradient, top: Value, immersed: ZeroFlux
└── data: 22×22×22 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, -2:19) with eltype Float64 with indices -2:19×-2:19×-2:19
    └── max=0.0, min=0.0, mean=0.0
```

Notice that the specified non-default boundary conditions have been applied at
top and bottom of both `model.velocities.u` and `model.tracers.c`.

## Immersed boundary conditions

Immersed boundary conditions are supported experimentally. A no-slip boundary condition is specified
by writing

```jldoctest
julia> underlying_grid = RectilinearGrid(size=(32, 32, 16), x=(-3, 3), y=(-3, 3), z=(0, 1), topology=(Periodic, Periodic, Bounded));

julia> hill(x, y) = 0.1 + 0.1 * exp(-x^2 - y^2)
hill (generic function with 1 method)

julia> grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(hill))
32×32×16 ImmersedBoundaryGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo:
├── immersed_boundary: GridFittedBottom{OffsetArrays.OffsetMatrix{Float64, Matrix{Float64}}}
├── underlying_grid: 32×32×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [-3.0, 3.0) regularly spaced with Δx=0.1875
├── Periodic y ∈ [-3.0, 3.0) regularly spaced with Δy=0.1875
└── Bounded  z ∈ [0.0, 1.0]  regularly spaced with Δz=0.0625

julia> velocity_bcs = FieldBoundaryConditions(immersed=ValueBoundaryCondition(0));

julia> model = NonhydrostaticModel(; grid, boundary_conditions=(u=velocity_bcs, v=velocity_bcs, w=velocity_bcs));

julia> model.velocities.w.boundary_conditions.immersed
ImmersedBoundaryCondition:
├── west: ValueBoundaryCondition: 0
├── east: ValueBoundaryCondition: 0
├── south: ValueBoundaryCondition: 0
├── north: ValueBoundaryCondition: 0
├── bottom: Nothing
└── top: Nothing
```

An `ImmersedBoundaryCondition` encapsulates boundary conditions on each potential boundary-facet
of a boundary-adjcent cell. Boundary conditions on specific faces of immersed-boundary-adjacent
cells may also be specified by manually building `ImmersedBoundaryCondition`:

```jldoctest
julia> bottom_drag_bc = ImmersedBoundaryCondition(bottom=ValueBoundaryCondition(0))
ImmersedBoundaryCondition:
├── west: Nothing
├── east: Nothing
├── south: Nothing
├── north: Nothing
├── bottom: ValueBoundaryCondition: 0
└── top: Nothing

julia> velocity_bcs = FieldBoundaryConditions(immersed=bottom_drag_bc)
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── east: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── south: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── north: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── bottom: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── top: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
└── immersed: ImmersedBoundaryCondition with west=Nothing, east=Nothing, south=Nothing, north=Nothing, bottom=Value, top=Nothing
```


