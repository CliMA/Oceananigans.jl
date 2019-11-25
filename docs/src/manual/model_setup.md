# Model setup

This section describes all the options and features that can be used to set up a model. For more detailed information
consult the API documentation.

Each structure covered in this section can be constructed and passed to the `Model` constructor. For examples of model
construction, see the examples. The verification experiments provide more advanced examples.

## Architecture
Passing `architecture = CPU()` or `architecture = GPU()` to the `Model` constructor will determine whether the model
is time stepped on a CPU or GPU.

Ideally a set up or simulation script does not need to be modified to run on a GPU but we are still smoothing out
rough edges. Generally the CPU wants `Array` objects while the GPU wants `CuArray` objects.

!!! tip "Running on GPUs"
    If you are having issues with running Oceananigans on a GPU, please
    [open an issue](https://github.com/climate-machine/Oceananigans.jl/issues/new) and we'll do our best to help out.

## Number type
Passing `float_type=Float64` or `float_type=Float32` to the `Model` constructor causes the model to store all numbers
with 64-bit or 32-bit floating point precision.

<!-- !!! note "Avoiding mixed-precision operations"
     -->

!!! warning "Effect of floating point precision on simulation accuracy"
    While we run many tests with both `Float32` and `Float64` it is not clear whether `Float32` is precise enough to
    provide similar accuracy in all use cases. If accuracy is a concern, stick to `Float64`.

    We will be actively investigating the possibility of using lower precision floating point numbers such as `Float32`
    and `Float16` for fluid dynamics as well as the use of alternative number types such as Posits and Sonums.

## Grids
Currently only a regular Cartesian grid with constant grid spacings is supported. The spacing can be different for each
dimension.

When constructing a `RegularCartesianGrid` the number of grid points (or size of the grid) must be passed a tuple along
with the physical length of each dimension.

A regular Cartesian grid with $N_x \times N_y \times N_z = 64 \times 32 \times 16$ grid points and a length of
$L_x = 200$ meters, $L_y = 100$ meters, and $L_z = 100$ meters is constructed using
```@example
grid = RegularCartesianGrid(size=(64, 32, 16), length=(200, 100, 100))
```

!!! info "Default domain"
    By default $x \in [0, L_x]$, $y \in [0, L_y]$, and $z \in [-L_z, 0]$ which is common for oceanographic applications.

### Specifying the domain
To specify a different domain, the `x`, `y`, and `z` keyword arguments can be used instead of `length`. For example,
to use the domain $x \in [-100, 100]$ meters, $y \in [-50, 50]$ meters, and $z \in [0, 100]$ meters
```@example
grid = RegularCartesianGrid(size=(64, 32, 16), x=(-100, 100), y=(-50, 50), z=(0, 100))
```

### Two-dimensional grids
Two-dimensional grids can be constructed by setting the number of grid points along the flat dimension to be 1. A
two-dimensional grid in the $xz$-plane can be constructed using
```@example
grid = RegularCartesianGrid(size=(64, 1, 16), length=(200, 1, 100))
```

In this case the length of the $y$ dimension must be specified but does not matter so we just set it to 1.

2D grids can be used to simulate $xy$, $xz$, and $yz$ planes.

### One-dimensional grids
One-dimensional grids can be constructed in a similar manner, most commonly used to set up vertical column models. For
example, to set up a 1D model with $N_z$ grid points
```@example
grid = RegularCartesianGrid(size=(1, 1, 90), length=(1, 1, 1000))
```

!!! warning "One-dimensional horizontal models"
    We only test one-dimensional vertical models and cannot guarantee that one-dimensional horizontal models will work
    as expected.

## Clock
The clock holds the current iteration number and time. By default the model starts at iteration number 0 and time 0
```@example
clock = Clock(0, 0)
```
but can be modified if you wish to start the model clock at some other time. If you want iteration 0 to correspond to
$t = 3600$ seconds, then you can construct
```#@example
clock = Clock(0, 3600)
```
and pass it to the model.

## Coriolis
The Coriolis option determines whether the fluid experiences the effect of the Coriolis force, or rotation. Currently
three options are available: no rotation, $f$-plane, and $\beta$-plane.

!!! info "Coriolis vs. rotation"
    If you are wondering why this option is called "Coriolis" it is because rotational effects could include the
    Coriolis and centripetal forces, both of which arise in non-inertial reference frames. But here the model only
    considers the Coriolis force.

To use no rotation, pass
```@example
coriolis = nothing
```

### $f$-plane

To set up an $f$-plane with, for example, rotation rate $f = 10^{-4} \text{s}^{-1}$
```@example
coriolis = FPlane(f=1e-4)
```

An $f$-plane can also be specified at some latitude on a spherical planet with a planetary rotation rate. For example,
to specify an $f$-plane at a latitude of $\varphi = 45°\text{N}$ on Earth which has a rotation rate of
$\Omega = 7.292115 \times 10^{-5} \text{s}^{-1}$
```@example
coriolis = FPlane(rotation_rate=7.292115e-5, latitude=45)
```
in which case the value of $f$ is given by $2\Omega\sin\varphi$.

### $\beta$-plane
To set up a $\beta$-plane the background rotation rate $f_0$ and the $\beta$ parameter must be specified. For example,
a $\beta$-plane with $f_0 = 10^{-4} \text{s}^{-1}$ and $\beta = 1.5 \times 10^{-11} \text{s}^{-1}\text{m}^{-1}$ can be
set up with
```@example
coriolis = BetaPlane(f₀=1e-4, β=1.5e-11)
```

Alternatively, a $\beta$-plane can also be set up at some latitude on a spherical planet with a planetary rotation rate
and planetary radius. For example, to specify a $\beta$-plane at a latitude of $\varphi = 10\degree{S}$ on Earth
which has a rotation rate of $\Omega = 7.292115 \times 10^{-5} \text{s}^{-1}$ and a radius of $R = 6,371 \text{km}$
```@example
coriolis = BetaPlane(rotation_rate=7.292115e-5, latitude=-10, radius=6371e3)
```
in which case $f_0 = 2\Omega\sin\varphi$ and $\beta = 2\Omega\cos\varphi / R$.

## Tracers

The tracers to be advected around can be specified via a list of symbols. By default the model evolves temperature and
salinity
```@example
tracers = (:T, :S)
```
but any number of arbitrary tracers can be appended to this list. For example, to evolve quantities $C_1$, CO₂, and
nitrogen as passive tracers you could set them up as
```@example
tracers = (:T, :S, :C₁, :CO₂, :nitrogen)
```

!!! info "Active vs. passive tracers"
    An active tracer typically denotes a tracer quantity that affects the fluid dynamics through buoyancy. In the ocean
    temperature and salinity are active tracers. Passive tracers, on the other hand, typically do not affect the fluid
    dynamics are are _passively_ advected around by the flow field.

## Buoyancy and equation of state
The buoyancy option selects how buoyancy is treated. There are currently three options:
1. No buoyancy (and no gravity).
2. Evolve buoyancy as a tracer.
3. _Seawater buoyancy_: evolve temperature $T$ and salinity $S$ as tracers with a value for the gravitational
   acceleration $g$ and an appropriate equation of state.

### No buoyancy
To turn off buoyancy (and gravity) simply pass
```@example
buoyancy = nothing
```
to the `Model` constructor. In this case, you will probably also want to explicitly specify which tracers to evolve.
In particular, you probably will not want to evolve temperature and salinity, which are included by default. To specify
no tracers, also pass
```@example
tracers = ()
```
to the `Model` constructor.

### Buoyancy as a tracer
To directly evolve buoyancy as a tracer simply pass
```@example
buoyancy = BuoyancyTracer()
```
to the `Model` constructor. Buoyancy `:b` must be included as a tracer, for example, by also passing
```@example
tracers = (:b)
```

### Seawater buoyancy
To evolve temperature $T$ and salinity $S$ and diagnose the buoyancy, you can pass
```@example
buoyancy = SeawaterBuoyancy()
```
which is also the default. Without any options specified, a value of $g = 9.80665 \; \text{m/s}^2$ is used for the
gravitational acceleration (corresponding to [standard gravity](https://en.wikipedia.org/wiki/Standard_gravity)) along
with a linear equation of state with thermal expansion and haline contraction coefficients suitable for water.

If, for example, you wanted to simulate fluids on another planet such as Europa where $g = 1.3 \; \text{m/s}^2$, then
use
```@example
buoyancy = SeawaterBuoyancy(gravitational_acceleration=1.3)
```

#### Linear equation of state
To use non-default thermal expansion and haline contraction coefficients, say
$\alpha = 2 \times 10^{-3} \; \text{K}^{-1}$ and $\beta = 5 \times 10{-4} \text{ppt}^{-1}$ corresponding to some other
fluid, then use

```@example
buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState(α=1.67e-4, β=7.80e-4))
```

#### Idealized nonlinear equation of state
Instead of a linear equation of state, an idealized equation of state as described by Roquet et al. (2015) may be
specified. See [`RoquetIdealizedNonlinearEquationOfState`](@ref RoquetIdealizedNonlinearEquationOfState).


## Boundary conditions
A boundary condition is applied to each field, dimension, and endpoint. There are left and right (or bottom and top)
boundary conditions for each of the x, y, and z dimensions so each field is associated with 6 boundary conditions. Each
of these boundary conditions may be specified individually. Each boundary condition can be specified via a constant
value, an array, or a function.

!!! warning "Consistent boundary conditions"
    Be careful to ensure that you don't set up a model with inconsistent boundary conditions. For example, periodic
    boundary conditions should remain imposed on all fields and endpoints for periodic dimensions, and velocities
    normal to a wall (e.g. vertical velocity w with walls at the top and bottom) must have no-penetration boundary
    conditions.

### Boundary condition structures

### Types of boundary conditions
1. Periodic
2. Flux
3. Value (or Dirchlet)
4. Gradient (or Neumann)
5. No-penetration

### Specifying boundary conditions with functions

#### Using functions of _(x, y, z, t)_

## Forcing functions

## Parameters

## Turbulent diffusivity closures and large eddy simulation models

## Diagnostics

## Output writers

## Setting initial conditions
Initial conditions are imposed after model construction. This can be easily done using the the `set!` function, which
allows the setting of initial conditions using constant values, arrays, or functions.

<!-- ## Model construction

!!! info "Units"
    By default the model assumes SI units. To set up a model with dimensionless units, see `NonDimensionalModel`. -->
