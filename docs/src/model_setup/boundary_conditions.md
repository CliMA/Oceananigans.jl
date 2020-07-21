# Boundary conditions

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
3. A set of three `CoordinateBoundaryConditions` specifying the boundary conditions along the x, y, and z dimensions
   for a single field are grouped into a [`FieldBoundaryConditions`](@ref) named tuple.
4. A set of `FieldBoundaryConditions`, up to one for each field, are grouped together into a named tuple and passed
   to the model constructor.

Boundary conditions are defined at model construction time by passing a named tuple of `FieldBoundaryConditions`
specifying non-default boundary conditions for fields such as velocities ($u$, $v$, $w$) and tracers. Not passing
in a `FieldBoundaryConditions` for one field means it gets the default boundary conditions.

See the sections below for more details. The examples and verification experiments also provide examples for setting up
different kinds of boundary conditions.

## Creating individual boundary conditions

```@meta
DocTestSetup = quote
   using Random
   using Oceananigans
   using Oceananigans.Fields
   using Oceananigans.BoundaryConditions
   Random.seed!(1234)
end
```

Some examples of creating individual boundary conditions:

1. A constant `Value` (Dirchlet) boundary condition, perhaps representing a constant temperature at some boundary.

   ```jldoctest
   julia> constant_T_bc = ValueBoundaryCondition(20.0)
   BoundaryCondition: type=Value, condition=20.0
   ```

2. A constant flux boundary condition, perhaps representing a constant wind stress at some boundary such as the ocean
   surface.

   ```jldoctest
   julia> ρ₀ = 1027;  # Reference density [kg/m³]
   julia> τₓ = 0.08;  # Wind stress [N/m²]
   julia> wind_stress_bc = FluxBoundaryCondition(τₓ/ρ₀)
   BoundaryCondition: type=Flux, condition=7.789678675754625e-5
   ```

3. A spatially varying (white noise) cooling flux to be imposed at some boundary. Note that the boundary condition
   is given by the array `Q` here. When running on the GPU, `Q` must be converted to a `CuArray`.

   ```jldoctest
   julia> Nx = Ny = 16;  # Number of grid points.
   julia> ρ₀ = 1027;  # Reference density [kg/m³]
   julia> cₚ = 4000;  # Heat capacity of water at constant pressure [J/kg/K]
   julia> Q  = randn(Nx, Ny) ./ (ρ₀ * cₚ);
   julia> white_noise_T_bc = FluxBoundaryCondition(Q)
   BoundaryCondition: type=Flux, condition=16×16 Array{Float64,2}
   ```

## Specifying boundary conditions with functions

If you need maximum flexibility you can also specify the boundary condition via a function. There are a few different
interfaces for doing this depending on whether you want access to the grid indices `i, j, k` or grid coordinates
`x, y, z` in the function signature, or whether you need to make use of parameters such as length scales or scaling
exponents in the function.

!!! info "Performance of functions in boundary conditions"
    For performance reasons, you should define all functions used in boundary conditions as inline functions via the
    `@inline` macro. If any arrays are accessed within the function, disabling bounds-checking with `@inbounds` will
    also speed things up. These are important considerations as these functions will be called many times every
    time step.

### Boundary condition functions with grid index access

Boundary condition functions with grid index `i, j, k` access, for example for the z dimension, must be specified
with the signature

```julia
f(i, j, grid, clock, state)
```

where `i, j` is the grid index, `grid` is `model.grid`, `clock` is the `model.clock`, and `state` is a named tuple
containing `state.velocities`, `state.tracers`, and `state.diffusivities`. The signature is similar for x and y
boundary conditions expect that `i, j` is replaced with `j, k` and `i, k` respectively.

```jldoctest
julia> @inline linear_drag(i, j, grid, clock, state) = @inbounds -0.2*state.velocities.u[i, j, 1]
julia> u_bottom_bc = FluxBoundaryCondition(linear_drag)
BoundaryCondition: type=Flux, condition=linear_drag(i, j, grid, clock, state) in Main at REPL[12]:1
```

Instead of hard-coding in the drag coefficient of 0.2, we may want to turn it, and other magic numbers, into parameters.
We would then use a [`ParameterizedBoundaryCondition`](@ref) and use the signature

```julia
f(i, j, grid, clock, state, parameters)
```

which would convert the above example into

```jldoctest
julia> C = 0.2;  # drag coefficient
julia> parameters = (C=C,);
julia> @inline linear_drag(i, j, grid, clock, state, parameters) =
           @inbounds - parameters.C * state.velocities.u[i, j, 1];
julia> u_bottom_bc = ParameterizedBoundaryCondition(Flux, linear_drag, parameters)
BoundaryCondition: type=Flux, condition=linear_drag(i, j, grid, clock, state, parameters) in Main at REPL[4]:1
```

### Boundary condition functions with grid coordinate access

To define boundary condition function with grid coordinate access, you should use a [`BoundaryFunction`](@ref). For example
for the z dimension, you must use the signature

```julia
f(x, y, t)
```

where `x, y` are the grid coordinates and `t` is the `model.clock.time`. The signature is similar for x and y
boundary conditions expect that `x, y` is replaced with `y, z` and `x, z` respectively.

```jldoctest
julia> surface_flux(x, y, t) = cos(2π*x) * cos(t);
julia> top_tracer_boundary_function = BoundaryFunction{:z, Cell, Cell}(surface_flux);
julia> top_tracer_bc = FluxBoundaryCondition(top_tracer_boundary_function)
BoundaryCondition: type=Flux, condition=surface_flux(x, y, t) in Main at REPL[3]:1
```

To add user-defined parameters such as a length-scale or scaling exponent, for example to a boundary condition function
in the z dimension, you would use the signature

```julia
f(x, y, t, parameters)
```

where `parameters` can be any structure that is passed to the [`BoundaryFunction`](@ref).

```jldoctest
julia> params = (k=4π, ω=3.0);
julia> flux_func(x, y, t, p) = cos(p.k * x) * cos(p.ω * t); # function with parameters
julia> parameterized_u_velocity_flux = BoundaryFunction{:z, Face, Cell}(flux_func, params);
julia> top_u_bc = BoundaryCondition(Flux, parameterized_u_velocity_flux)
BoundaryCondition: type=Flux, condition=flux_func(x, y, t, p) in Main at REPL[7]:1
```

## Specifying boundary conditions on a field

To, for example, create a set of horizontally periodic field boundary conditions

```jldoctest
julia> topology = (Periodic, Periodic, Bounded);
julia> grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=topology);
julia> T_bcs = TracerBoundaryConditions(grid,    top = ValueBoundaryCondition(20),
                                              bottom = GradientBoundaryCondition(0.01))
Oceananigans.FieldBoundaryConditions (NamedTuple{(:x, :y, :z)}), with boundary conditions
├── x: CoordinateBoundaryConditions{BoundaryCondition{Oceananigans.BoundaryConditions.Periodic,Nothing},BoundaryCondition{Oceananigans.BoundaryConditions.Periodic,Nothing}}
├── y: CoordinateBoundaryConditions{BoundaryCondition{Oceananigans.BoundaryConditions.Periodic,Nothing},BoundaryCondition{Oceananigans.BoundaryConditions.Periodic,Nothing}}
└── z: CoordinateBoundaryConditions{BoundaryCondition{Gradient,Float64},BoundaryCondition{Value,Int64}}
```

which will create a [`FieldBoundaryConditions`](@ref) object for temperature `T` appropriate for horizontally periodic
model configurations where the x and y boundary conditions are all periodic.

## Specifying model boundary conditions

A named tuple of [`FieldBoundaryConditions`](@ref) objects must be passed to the Model constructor specifying boundary
conditions on all fields. To, for example, impose non-default boundary conditions on the u-velocity and temperature

```jldoctest
julia> topology = (Periodic, Periodic, Bounded);
julia> grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=topology);
julia> u_bcs = UVelocityBoundaryConditions(grid,   top = ValueBoundaryCondition(+0.1),
                                                bottom = ValueBoundaryCondition(-0.1));
julia> T_bcs = TracerBoundaryConditions(grid,   top = ValueBoundaryCondition(20),
                                             bottom = GradientBoundaryCondition(0.01));
julia> model = Model(grid=grid, boundary_conditions=(u=u_bcs, T=T_bcs))
IncompressibleModel{CPU, Float64}(time = 0.000 s, iteration = 0) 
├── grid: RegularCartesianGrid{Float64, Periodic, Periodic, Bounded}(Nx=16, Ny=16, Nz=16)
├── tracers: (:T, :S)
├── closure: ConstantIsotropicDiffusivity{Float64,NamedTuple{(:T, :S),Tuple{Float64,Float64}}}
├── buoyancy: SeawaterBuoyancy{Float64,LinearEquationOfState{Float64},Nothing,Nothing}
└── coriolis: Nothing
```
