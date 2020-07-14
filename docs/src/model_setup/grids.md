# Grids

Currently only a regular Cartesian grid with constant grid spacings is supported. The spacing can be different for each
dimension.

When constructing a `RegularCartesianGrid` the number of grid points (or `size` of the grid) and the physical length
of each dimension (or `length` of the grid) must be passed as tuples.

A regular Cartesian grid with $N_x \times N_y \times N_z = 64 \times 32 \times 16$ grid points and an extent of
$L_x = 200$ meters, $L_y = 100$ meters, and $L_z = 100$ meters is constructed using

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest
grid = RegularCartesianGrid(size=(64, 32, 16), extent=(200, 100, 100))

# output

RegularCartesianGrid{Float64, Periodic, Periodic, Bounded}
                   domain: x ∈ [0.0, 200.0], y ∈ [0.0, 100.0], z ∈ [-100.0, 6.25]
                 topology: (Periodic, Periodic, Bounded)
  resolution (Nx, Ny, Nz): (64, 32, 16)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (3.125, 3.125, 6.25)
```

!!! info "Default domain"
    By default $x \in [0, L_x]$, $y \in [0, L_y]$, and $z \in [-L_z, 0]$ which is common for oceanographic applications.

## Specifying the grid's topology

You can also pass a tuple denoting the grid's topology. Each dimension can be either `Periodic`, `Bounded`, or `Flat`.
By default, the `RegularCartesianGrid` constructor assumes a horizontally periodic grid topology,
`topology = (Periodic, Periodic, Bounded)`. To specify a channel model that is periodic in the x-dimension and wall-bounded
in the y- and z-dimensions:

```jldoctest
grid = RegularCartesianGrid(topology=(Periodic, Bounded, Bounded),
                            size=(64, 32, 16), extent=(200, 100, 100))

# output

RegularCartesianGrid{Float64, Periodic, Bounded, Bounded}
                   domain: x ∈ [0.0, 200.0], y ∈ [0.0, 103.125], z ∈ [-100.0, 6.25]
                 topology: (Periodic, Bounded, Bounded)
  resolution (Nx, Ny, Nz): (64, 32, 16)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (3.125, 3.125, 6.25)
```

## Specifying the domain

To specify a different domain, the `x`, `y`, and `z` keyword arguments can be used instead of `length`. For example,
to use the domain $x \in [-100, 100]$ meters, $y \in [0, 12.5]$ meters, and $z \in [0, \pi]$ meters

```jldoctest
grid = RegularCartesianGrid(size=(64, 32, 16), x=(-100, 100), y=(0, 12.5), z=(-π, π))

# output

RegularCartesianGrid{Float64, Periodic, Periodic, Bounded}
                   domain: x ∈ [-100.0, 100.0], y ∈ [0.0, 12.5], z ∈ [-3.141592653589793, 3.5342917352885173]
                 topology: (Periodic, Periodic, Bounded)
  resolution (Nx, Ny, Nz): (64, 32, 16)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (3.125, 0.390625, 0.39269908169872414)
```

## Two-dimensional grids

Two-dimensional grids can be constructed by setting the number of grid points along the flat dimension to be 1. A
two-dimensional grid in the $xz$-plane can be constructed using

```jldoctest
grid = RegularCartesianGrid(topology=(Periodic, Flat, Bounded),
                            size=(64, 1, 16), extent=(200, 1, 100))

# output

RegularCartesianGrid{Float64, Periodic, Flat, Bounded}
                   domain: x ∈ [0.0, 200.0], y ∈ [0.0, 1.0], z ∈ [-100.0, 6.25]
                 topology: (Periodic, Flat, Bounded)
  resolution (Nx, Ny, Nz): (64, 1, 16)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (3.125, 1.0, 6.25)
```

In this case the length of the $y$ dimension must be specified but does not matter so we just set it to 1.

2D grids can be used to simulate $xy$, $xz$, and $yz$ planes.

## One-dimensional grids

One-dimensional grids can be constructed in a similar manner, most commonly used to set up vertical column models. For
example, to set up a 1D model with $N_z$ grid points

```jldoctest
grid = RegularCartesianGrid(topology=(Flat, Flat, Bounded),
                            size=(1, 1, 90), extent=(1, 1, 1000))

# output

RegularCartesianGrid{Float64, Flat, Flat, Bounded}
                   domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [-1000.0, 11.111111111111086]
                 topology: (Flat, Flat, Bounded)
  resolution (Nx, Ny, Nz): (1, 1, 90)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (1.0, 1.0, 11.11111111111111)
```

The length of the $x$ and $y$ dimensions must be specified but do not matter.

!!! warning "One-dimensional horizontal models"
    We only test one-dimensional vertical models and cannot guarantee that one-dimensional horizontal models will work
    as expected. But please [open an issue](https://github.com/CLiMA/Oceananigans.jl/issues/new) if you're trying to
    use one-dimensional horizontal models and we'll do our best to help out.
