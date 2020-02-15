# Grids
Currently only a regular Cartesian grid with constant grid spacings is supported. The spacing can be different for each
dimension.

When constructing a `RegularCartesianGrid` the number of grid points (or `size` of the grid) and the physical length
of each dimension (or `length` of the grid) must be passed as tuples.

A regular Cartesian grid with $N_x \times N_y \times N_z = 64 \times 32 \times 16$ grid points and a length of
$L_x = 200$ meters, $L_y = 100$ meters, and $L_z = 100$ meters is constructed using
```@example
using Oceananigans # hide
grid = RegularCartesianGrid(size=(64, 32, 16), length=(200, 100, 100))
```

!!! info "Default domain"
    By default $x \in [0, L_x]$, $y \in [0, L_y]$, and $z \in [-L_z, 0]$ which is common for oceanographic applications.

## Specifying the grid's topology
You can also pass a tuple denoting the grid's topology. Each dimension can be either `Periodic`, `Bounded`, or `Flat`.
By default, the `RegularCartesianGrid` constructor assumes a horizontally periodic grid topology,
`topology = (Periodic, Periodic, Bounded)`. To specify a channel model that is periodic in the x-dimension and wall-bounded
in the y- and z-dimensions:
```@example
using Oceananigans # hide
grid = RegularCartesianGrid(size=(64, 32, 16), length=(200, 100, 100), topology=(Periodic, Bounded, Bounded))
```

## Specifying the domain
To specify a different domain, the `x`, `y`, and `z` keyword arguments can be used instead of `length`. For example,
to use the domain $x \in [-100, 100]$ meters, $y \in [-50, 50]$ meters, and $z \in [0, 100]$ meters
```@example
using Oceananigans # hide
grid = RegularCartesianGrid(size=(64, 32, 16), x=(-100, 100), y=(-50, 50), z=(0, 100))
```

## Two-dimensional grids
Two-dimensional grids can be constructed by setting the number of grid points along the flat dimension to be 1. A
two-dimensional grid in the $xz$-plane can be constructed using
```@example
using Oceananigans # hide
grid = RegularCartesianGrid(size=(64, 1, 16), length=(200, 1, 100), topology=(Periodic, Flat, Bounded))
```

In this case the length of the $y$ dimension must be specified but does not matter so we just set it to 1.

2D grids can be used to simulate $xy$, $xz$, and $yz$ planes.

## One-dimensional grids
One-dimensional grids can be constructed in a similar manner, most commonly used to set up vertical column models. For
example, to set up a 1D model with $N_z$ grid points
```@example
using Oceananigans # hide
grid = RegularCartesianGrid(size=(1, 1, 90), length=(1, 1, 1000), topology=(Flat, Flat, Bounded))
```
The length of the $x$ and $y$ dimensions must be specified but do not matter.

!!! warning "One-dimensional horizontal models"
    We only test one-dimensional vertical models and cannot guarantee that one-dimensional horizontal models will work
    as expected.
