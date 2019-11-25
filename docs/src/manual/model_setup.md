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
To set up a $\beta$-plane the background rotation rate $f\_0$ and the $\beta$ parameter must be specified. For example,
a $\beta$-plane with $f\_0 = 10^{-4} \text{s}^{-1}$ and $\beta = 1.5 \times 10^{-11} \text{s}^{-1}\text{m}^{-1}$ can be
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
in which case $f\_0 = 2\Omega\sin\varphi$ and $\beta = 2\Omega\cos\varphi / R$.

## Model

!!! info "Units"
    By default the model assumes SI units. To set up a model with dimensionless units, see `NonDimensionalModel`.
