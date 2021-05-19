# [Boundary conditions](@id model_step_bcs)

A boundary condition is applied to each field, dimension, and endpoint. There are left and right boundary conditions
for each of the x, y, and z dimensions so each field has 6 boundary conditions. Each of these boundary conditions may
be specified individually. Each boundary condition can be specified via a constant value, an array, or a function.

The left and right boundary conditions associated with the x-dimension are called west and east, respectively. For the
y-dimension, left and right are called south and north. For the z-dimension, left and right are called bottom and top.

See [Numerical implementation of boundary conditions](@ref numerical_bcs) for more details.

## Types of boundary conditions

1. [`Periodic`](@ref)
2. [`Flux`](@ref)
3. [`Value`](@ref) (Dirchlet)
4. [`Gradient`](@ref) (Neumann)
5. [`NormalFlow`](@ref)

To construct a `Flux` boundary condition use the `FluxBoundaryCondition` constructor and so on.

Notice that open boundary conditions and radiation boundary conditions can be imposed via flux or value boundary
conditions defined by a function or array. Or alternatively, through a forcing function if more flexibility is
desired.

## Default boundary conditions

By default, periodic boundary conditions are applied on all fields along periodic dimensions. Otherwise tracers
get no-flux boundary conditions and velocities get free-slip and no normal flow boundary conditions.

## Boundary condition structures

Oceananigans uses a hierarchical structure to express boundary conditions.

1. A [`BoundaryCondition`](@ref) is associated with every field, dimension, and endpoint.
2. Boundary conditions specifying the condition at the left and right endpoints are
   grouped into [`CoordinateBoundaryConditions`](@ref).
3. A set of three `CoordinateBoundaryConditions` specifying the boundary conditions along the ``x``-, ``y``-,
   and ``z``-dimensions.
   for a single field are grouped into a [`FieldBoundaryConditions`](@ref) `NamedTuple`.
4. A set of `FieldBoundaryConditions`, up to one for each field, are grouped into a `NamedTuple` and passed
   to the model constructor.

Boundary conditions are defined at model construction time by passing a `NamedTuple` of `FieldBoundaryConditions`
specifying non-default boundary conditions for fields such as velocities (``u``, ``v``, ``w``) and tracers.
Fields for which boundary conditions are not specified are assigned a default boundary conditions.
Note that default boundary conditions depend on the grid topology.

A few illustrations are provided below. See the validation experiments and examples for
further illustrations of boundary condition specification.

## Creating individual boundary conditions with `BoundaryCondition`

```@meta
DocTestSetup = quote
   using Oceananigans
   using Oceananigans.BoundaryConditions

   using Random
   Random.seed!(1234)
end
```

Boundary conditions may be specified with constants, functions, or arrays.
In this section we illustrate usage of the different [`BoundaryCondition`](@ref) constructors.

### 1. Constant `Value` (Dirchlet) boundary condition

```jldoctest
julia> constant_T_bc = ValueBoundaryCondition(20.0)
BoundaryCondition: type=Value, condition=20.0
```

A constant [`Value`](@ref) boundary condition can be used to specify constant tracer (such as temperature),
or a constant _tangential_ velocity component at a boundary. Note that boundary conditions on the
_normal_ velocity component must use the [`NormalFlow`](@ref) boundary condition type.

Finally, note that `ValueBoundaryCondition(condition)` is an alias for `BoundaryCondition(Value, condition)`.

### 2. Constant `Flux` boundary condition

```jldoctest
julia> ρ₀ = 1027;  # Reference density [kg/m³]

julia> τₓ = 0.08;  # Wind stress [N/m²]

julia> wind_stress_bc = FluxBoundaryCondition(-τₓ/ρ₀)
BoundaryCondition: type=Flux, condition=-7.789678675754625e-5
```

A constant [`Flux`](@ref) boundary condition can be imposed on tracers and tangential velocity components
can can be used, for example, to specify cooling, heating, evaporation, or wind stress at the ocean surface.

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
BoundaryCondition: type=Flux, condition=surface_flux(x, y, t) in Main at none:1
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

julia> top_u_bc = BoundaryCondition(Flux, wind_stress, parameters=(k=4π, ω=3.0, τ=1e-4))
BoundaryCondition: type=Flux, condition=wind_stress(x, y, t, p) in Main at none:1
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
julia> @inline linear_drag(x, y, z, t, u) = - 0.2 * u
linear_drag (generic function with 1 method)

julia> u_bottom_bc = FluxBoundaryCondition(linear_drag, field_dependencies=:u)
BoundaryCondition: type=Flux, condition=linear_drag(x, y, z, t, u) in Main at none:1
```

### 6. 'Field-dependent' boundary conditions with parameters

When boundary conditions depends on fields _and_ parameters, their functions take the form

```jldoctest
julia> @inline quadratic_drag(x, y, z, t, u, v, drag_coeff) = - drag_coeff * u * sqrt(u^2 + v^2)
quadratic_drag (generic function with 1 method)

julia> u_bottom_bc = FluxBoundaryCondition(quadratic_drag, field_dependencies=(:u, :v), parameters=1e-3)
BoundaryCondition: type=Flux, condition=quadratic_drag(x, y, z, t, u, v, drag_coeff) in Main at none:1
```

Put differently, field dependencies follow `ξ, η, t` come first in the function signature,
which are in turn followed by `parameters`.

### 7. Discrete-form boundary condition with parameters

Discrete field data may also be accessed directly from boundary condition functions
using the `discrete_form`. For example:

```jldoctest
@inline filtered_drag(i, j, grid, clock, model_fields) =
   @inbounds - 0.05 * (model_fields.u[i-1, j, 1] + 2 * model_fields.u[i, j, 1] + model_fields.u[i-1, j, 1])

u_bottom_bc = FluxBoundaryCondition(filtered_drag, discrete_form=true)

# output
BoundaryCondition: type=Flux, condition=filtered_drag(i, j, grid, clock, model_fields) in Main at none:1
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

julia> u_bottom_bc = BoundaryCondition(Flux, linear_drag, discrete_form=true, parameters=Cd)
BoundaryCondition: type=Flux, condition=linear_drag(i, j, grid, clock, model_fields, Cd) in Main at none:1
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
BoundaryCondition: type=Flux, condition=16×16 Matrix{Float64}
```

When running on the GPU, `Q` must be converted to a `CuArray`.

## Building boundary conditions on a field

To create, for example, a set of horizontally-periodic field boundary conditions, write

```jldoctest
julia> topology = (Periodic, Periodic, Bounded);

julia> grid = RegularRectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=topology);

julia> T_bcs = TracerBoundaryConditions(grid,    top = ValueBoundaryCondition(20),
                                              bottom = GradientBoundaryCondition(0.01))
Oceananigans.FieldBoundaryConditions (NamedTuple{(:x, :y, :z)}), with boundary conditions
├── x: CoordinateBoundaryConditions{BoundaryCondition{Oceananigans.BoundaryConditions.Periodic, Nothing}, BoundaryCondition{Oceananigans.BoundaryConditions.Periodic, Nothing}}
├── y: CoordinateBoundaryConditions{BoundaryCondition{Oceananigans.BoundaryConditions.Periodic, Nothing}, BoundaryCondition{Oceananigans.BoundaryConditions.Periodic, Nothing}}
└── z: CoordinateBoundaryConditions{BoundaryCondition{Gradient, Float64}, BoundaryCondition{Value, Int64}}
```

`T_bcs` is a [`FieldBoundaryConditions`](@ref) object for temperature `T` appropriate
for horizontally periodic grid topologies.
The default `Periodic` boundary conditions in ``x`` and ``y`` are inferred from the `topology` of `grid`.

For ``u``, ``v``, and ``w``, use the
`UVelocityBoundaryConditions`
`VVelocityBoundaryConditions`, and
`WVelocityBoundaryConditions` constructors, respectively.

## Specifying model boundary conditions

To specify non-default boundary conditions, a named tuple of [`FieldBoundaryConditions`](@ref) objects is
passed to the keyword argument `boundary_conditions` in the [`IncompressibleModel`](@ref) constructor.
The keys of `boundary_conditions` indicate the field to which the boundary condition is applied.
Below, non-default boundary conditions are imposed on the ``u``-velocity and temperature.

```jldoctest
julia> topology = (Periodic, Periodic, Bounded);

julia> grid = RegularRectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=topology);

julia> u_bcs = UVelocityBoundaryConditions(grid, top = ValueBoundaryCondition(+0.1),
                                              bottom = ValueBoundaryCondition(-0.1));

julia> T_bcs = TracerBoundaryConditions(grid, top = ValueBoundaryCondition(20),
                                           bottom = GradientBoundaryCondition(0.01));

julia> model = IncompressibleModel(grid=grid, boundary_conditions=(u=u_bcs, T=T_bcs))
IncompressibleModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=16, Ny=16, Nz=16)
├── tracers: (:T, :S)
├── closure: Nothing
├── buoyancy: SeawaterBuoyancy{Float64, LinearEquationOfState{Float64}, Nothing, Nothing}
└── coriolis: Nothing

julia> model.velocities.u
Field located at (Face, Center, Center)
├── data: OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}, size: (16, 16, 16)
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=16, Ny=16, Nz=16)
└── boundary conditions: x=(west=Periodic, east=Periodic), y=(south=Periodic, north=Periodic), z=(bottom=Value, top=Value)

julia> model.tracers.T
Field located at (Center, Center, Center)
├── data: OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}, size: (16, 16, 16)
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=16, Ny=16, Nz=16)
└── boundary conditions: x=(west=Periodic, east=Periodic), y=(south=Periodic, north=Periodic), z=(bottom=Gradient, top=Value)
```

Notice that the specified non-default boundary conditions have been applied at
top and bottom of both `model.velocities.u` and `model.tracers.T`.
