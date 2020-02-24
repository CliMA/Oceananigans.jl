# Boundary conditions
A boundary condition is applied to each field, dimension, and endpoint. There are left and right boundary conditions
for each of the x, y, and z dimensions so each field is associated with 6 boundary conditions. Each of these boundary
conditions may be specified individually. Each boundary condition can be specified via a constant value, an array, or
a function.

The left and right boundary conditions associated with the x-dimension are called west and east, respectively. For the
y-dimension, left and right are called south and north. For the z-dimension, left and right are called bottom and top.

See [Numerical implementation of boundary conditions](@ref numerical_bcs) for more details.

## Types of boundary conditions
1. [`Periodic`](@ref Periodic)
2. [`Flux`](@ref Flux)
3. [`Value`](@ref Value)
4. [`Gradient`](@ref Gradient)
5. [`No-penetration`](@ref NoPenetration)

Notice that open boundary conditions and radiation boundary conditions can be imposed via flux or value boundary
conditions defined by a function or array. Or alternatively, through a forcing function which is more flexible.

## Default boundary conditions
By default, periodic boundary conditions are applied on all fields along periodic dimensions. Otherwise tracers
get no-flux boundary conditions and velocities get free-slip and no-penetration boundary conditions.

## Boundary condition structures
Oceananigans uses a hierarchical structure to expressing boundary conditions.
1. A [`BoundaryCondition`](@ref) is associated with every field, dimension, and endpoint.
2. Boundary conditions specifying the condition at the left and right endpoints are
   grouped into [`CoordinateBoundaryConditions`](@ref).
3. A set of three `CoordinateBoundaryConditions` specifying the boundary conditions along the x, y, and z dimensions
   for a single field are grouped into a [`FieldBoundaryConditions`](@ref) named tuple.
4. A set of `FieldBoundaryConditions`, one for each field, are grouped together into a named tuple and passed to the
   `Model` constructor.

Boundary conditions are defined at model construction time by passing a named tuple of `FieldBoundaryConditions`
specifying boundary conditions on every field: velocities ($u$, $v$, $w$) and all tracers.

See the sections below for more details. The examples and verification experiments also provide examples for setting up
many different kinds of boundary conditions.

## Creating individual boundary conditions
Some examples of creating individual boundary conditions:

1. A constant Value (Dirchlet) boundary condition, perhaps representing a constant temperature at some boundary.
```@example
using Oceananigans # hide
constant_T_bc = ValueBoundaryCondition(20)
```

2. A constant flux boundary condition, perhaps representing a constant wind stress at some boundary such as the ocean
   surface.
```@example
using Oceananigans # hide
ρ₀ = 1027  # Reference density [kg/m³]
τₓ = 0.08  # Wind stress [N/m²]
wind_stress_bc = FluxBoundaryCondition(τₓ/ρ₀)
```

3. A spatially varying (white noise) cooling flux to be imposed at some boundary. Note that the boundary condition
   is given by the array `Q` here. When running on the GPU, `Q` must be converted to a `CuArray`.
```@example
using Oceananigans # hide
Nx = Ny = 16  # Number of grid points.

ρ₀ = 1027  # Reference density [kg/m³]
cₚ = 4000  # Heat capacity of water at constant pressure [J/kg/K]

Q  = randn(Nx, Ny) ./ (ρ₀ * cₚ)

white_noise_T_bc = FluxBoundaryCondition(Q)
```

## Specifying boundary conditions with functions
You can also specify the boundary condition via a function. For z boundary conditions the function will be called with
the signature
```
f(i, j, grid, time, iteration, U, C, params)
```
where `i, j` is the grid index, `grid` is `model.grid`, `time` is the `model.clock.time`, `iteration` is the
`model.clock.iteration`, `U` is the named tuple `model.velocities`, `C` is the named tuple `C.tracers`, and `params` is
the user-defined `model.parameters`. The signature is similar for x and y boundary conditions expect that `i, j` is
replaced with `j, k` and `i, k` respectively.

We can add a fourth example now:
4. A spatially varying and time-dependent heating representing perhaps a localized source of heating modulated by a
   diurnal cycle.
```@example
using Oceananigans # hide
@inline Q(i, j, grid, t, I, U, C, params) = @inbounds exp(-(grid.xC[i]^2 + grid.yC[j]^2)) * sin(2π*t)
localized_heating_bc = FluxBoundaryCondition(Q)
```

!!! info "Performance of functions in boundary conditions"
    For performance reasons, you should define all functions used in boundary conditions as inline functions via the
    `@inline` macro. If any arrays are accessed within the function, disabling bounds-checking with `@inbounds` will
    also speed things up.

## Specifying boundary conditions on a field
To, for example, create a set of horizontally periodic field boundary conditions
```@example
using Oceananigans # hide
topology = (Periodic, Periodic, Bounded)
grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), topology=topology)
T_bcs = TracerBoundaryConditions(grid,    top = ValueBoundaryCondition(20),
                                       bottom = GradientBoundaryCondition(0.01))
```
which will create a [`FieldBoundaryConditions`](@ref) object for temperature T appropriate for horizontally periodic
model configurations where the x and y boundary conditions are all periodic.

## Specifying model boundary conditions
A named tuple of [`FieldBoundaryConditions`](@ref) objects must be passed to the Model constructor specifying boundary
conditions on all fields. To, for example, impose non-default boundary conditions on the u-velocity and temperature
```@example
using Oceananigans # hide
topology = (Periodic, Periodic, Bounded)
grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), topology=topology)

u_bcs = UVelocityBoundaryConditions(grid,   top = ValueBoundaryCondition(+0.1),
                                         bottom = ValueBoundaryCondition(-0.1))
T_bcs = TracerBoundaryConditions(grid,   top = ValueBoundaryCondition(20),
                                      bottom = GradientBoundaryCondition(0.01))

model_bcs = SolutionBoundaryConditions(u=u_bcs, T=T_bcs)

model = Model(grid=grid, boundary_conditions=model_bcs)
nothing # hide
```
