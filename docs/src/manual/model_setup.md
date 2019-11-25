# Model setup

This section describes all the options and features that can be used to set up a model. For more detailed information
consult the API documentation.

Each structure covered in this section can be constructed and passed to the `Model` constructor. For examples of model
construction, see the examples. The verification experiments provide more advanced examples.

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
By default $x \in [0, L_x]$, $y \in [0, L_y]$, and $z \in [-Lz, 0]$ which is common for oceanographic applications.

### Specifying the domain

To specify a different domain, the `x`, `y`, and `z` keyword arguments can be used instead of `length`. For example,
to use the domain $x \in [-100, 100]$, $y \in [-50, 50]$, and $z \in [0, 100]$
```@example
grid = RegularCartesianGrid(size=(64, 32, 16), x=(-100, 100), y=(-50, 50), z=(0, 100))
```

### Two-dimensional grids

Two-dimensional grids can be constructed by setting the number of grid points along the flat dimension to be 1. A
2D grid in the $xz$-plane can be constructed using
```@example
grid = RegularCartesianGrid(size=(64, 1, 16), length=(200, 1, 100))
```

In this case the length of the $y$ dimension must be specified but does not matter so we just set it to 1.

2D grids can be used to simulate $xy$, $xz$, and $yz$ planes.

### One-dimensional grids

One-dimensional grids can be constructed in a similar manner, although probably make the most sense as a vertical
column with $N_z$ grid points. For example,
```@example
grid = RegularCartesianGrid(size=(1, 1, 90), length=(1, 1, 1000))
```

## 
